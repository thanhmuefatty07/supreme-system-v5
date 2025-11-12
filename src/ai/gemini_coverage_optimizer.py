    async def _analyze_coverage(self, source_dir: str) -> float:
        """
        Analyze current test coverage by running pytest.
        
        Args:
            source_dir: Source directory to analyze
        """
        import subprocess
        
        try:
            # Run pytest with coverage
            logger.info(f"Running pytest coverage analysis on {source_dir}...")
            
            result = subprocess.run(
                ["pytest", f"--cov={source_dir}", "--cov-report=xml", "--cov-report=term", "-q"],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # Parse coverage.xml if it exists
            coverage_file = Path("coverage.xml")
            if coverage_file.exists():
                tree = ET.parse(coverage_file)
                root = tree.getroot()
                line_rate = float(root.attrib.get('line-rate', 0))
                
                logger.info(f"✅ Coverage analysis complete: {line_rate*100:.1f}%")
                return line_rate
            else:
                logger.warning("coverage.xml not found, returning 0")
                return 0.0
                
        except subprocess.TimeoutExpired:
            logger.error("Coverage analysis timeout after 300s")
            return 0.0
        except Exception as e:
            logger.error(f"Coverage analysis failed: {e}")
            return 0.0

    async def _identify_coverage_gaps(self, source_dir: str) -> List[CoverageGap]:
        """
        Identify specific coverage gaps from coverage report.
        """
        gaps = []
        try:
            # Parse coverage.xml
            coverage_file = Path("coverage.xml")
            if not coverage_file.exists():
                logger.warning("coverage.xml not found, running coverage first...")
                await self._analyze_coverage(source_dir)
            
            tree = ET.parse(coverage_file)
            root = tree.getroot()
            # Iterate through all classes/modules
            for package in root.findall(".//package"):
                for cls in package.findall(".//class"):
                    filename = cls.attrib.get('filename', '')
                    filepath = Path(source_dir) / filename
                    # Skip if file doesn't exist
                    if not filepath.exists():
                        continue
                    # Find uncovered lines
                    uncovered_lines = []
                    for line in cls.findall(".//line"):
                        hits = int(line.attrib.get('hits', 0))
                        if hits == 0:
                            line_number = int(line.attrib.get('number', 0))
                            uncovered_lines.append(line_number)
                    if not uncovered_lines:
                        continue
                    # Group consecutive uncovered lines
                    line_groups = self._group_consecutive_lines(uncovered_lines)
                    # Create coverage gaps for each group
                    for line_start, line_end in line_groups:
                        gap = await self._create_gap_with_context(
                            filepath, line_start, line_end
                        )
                        if gap:
                            gaps.append(gap)
            logger.info(f"✅ Identified {len(gaps)} coverage gaps")
            return gaps
        except Exception as e:
            logger.error(f"Coverage gap analysis failed: {e}")
            return []

    def _group_consecutive_lines(self, lines: List[int]) -> List[Tuple[int, int]]:
        """
        Group consecutive line numbers into ranges.
        """
        if not lines:
            return []
        lines = sorted(lines)
        groups = []
        start = lines[0]
        end = lines[0]
        for line in lines[1:]:
            if line == end + 1:
                end = line
            else:
                groups.append((start, end))
                start = line
                end = line
        groups.append((start, end))
        return groups

    async def _create_gap_with_context(
        self,
        filepath: Path,
        line_start: int,
        line_end: int
    ) -> Optional[CoverageGap]:
        """
        Create a CoverageGap object with code context.
        """
        try:
            # Read source file
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            # Extract context (20 lines before/after)
            context_start = max(0, line_start - 20)
            context_end = min(len(lines), line_end + 20)
            context_lines = lines[context_start:context_end]
            code_context = ''.join(context_lines)
            # Find function name
            function_name = self._find_function_name(lines, line_start)
            # Calculate complexity
            complexity = self._calculate_complexity(code_context)
            # Calculate priority
            priority = self._calculate_priority(complexity, function_name, code_context)
            return CoverageGap(
                file_path=str(filepath),
                line_start=line_start,
                line_end=line_end,
                code_context=code_context,
                function_name=function_name,
                complexity=complexity,
                priority=priority
            )
        except Exception as e:
            logger.warning(f"Failed to create gap for {filepath}:{line_start}: {e}")
            return None

    def _find_function_name(self, lines: List[str], line_num: int) -> str:
        """
        Find the function name containing the given line.
        """
        # Search backwards for function definition
        for i in range(line_num - 1, -1, -1):
            if i >= len(lines):
                continue
            line = lines[i]
            # Match function definitions
            match = re.match(r'\s*def\s+(\w+)\s*\(', line)
            if match:
                return match.group(1)
            # Match class definitions (if no function found)
            match = re.match(r'\s*class\s+(\w+)', line)
            if match:
                return f"{match.group(1)}_class"
        return "unknown_function"

    def _calculate_complexity(self, code: str) -> float:
        """
        Calculate cyclomatic complexity of code.
        """
        try:
            tree = ast.parse(code)
            complexity = 1
            for node in ast.walk(tree):
                # Count decision points
                if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
                elif isinstance(node, (ast.ExceptHandler,)):
                    complexity += 1
            return float(complexity)
        except:
            return 1.0

    def _calculate_priority(self, complexity: float, function_name: str, code: str) -> float:
        """
        Calculate priority score for a coverage gap.
        """
        priority = 0.5  # Base priority
        # Complexity bonus (higher complexity = higher priority)
        priority += min(complexity / 20, 0.3)
        # Critical function keywords
        critical_keywords = [
            'execute', 'trade', 'order', 'position', 'risk',
            'emergency', 'liquidation', 'close', 'stop', 'loss'
        ]
        if any(kw in function_name.lower() for kw in critical_keywords):
            priority += 0.2
        # Error handling bonus
        if any(keyword in code for keyword in ['try:', 'except', 'raise', 'assert']):
            priority += 0.15
        # Async bonus (async functions often critical)
        if 'async def' in code:
            priority += 0.1
        return min(priority, 1.0)
