#!/usr/bin/env python3
"""
Enterprise-Grade Coverage Instrumentation System
Bypasses sys.modules mocking limitations with AST-based instrumentation
"""

import ast
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Set, List
import json
import time
import threading

class EnterpriseCoverageInstrumenter(ast.NodeTransformer):
    """AST-based code coverage instrumentation that works with sys.modules mocking"""

    def __init__(self, module_name: str):
        self.module_name = module_name
        self.executed_lines: Set[int] = set()
        self.lock = threading.Lock()

    def visit_Stmt(self, node: ast.stmt) -> ast.stmt:
        """Instrument all statements with coverage tracking"""
        if hasattr(node, 'lineno') and node.lineno > 0:
            # Insert coverage tracking call before the statement
            coverage_call = ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='_enterprise_coverage_tracker', ctx=ast.Load()),
                        attr='track_line',
                        ctx=ast.Load()
                    ),
                    args=[
                        ast.Constant(value=self.module_name),
                        ast.Constant(value=node.lineno)
                    ],
                    keywords=[]
                )
            )

            # Return both the tracking call and the original statement
            return [coverage_call, node]

        return node

class EnterpriseCoverageTracker:
    """Global coverage tracking system"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.coverage_data: Dict[str, Set[int]] = {}
            self.lock = threading.Lock()

    def track_line(self, module_name: str, line_no: int):
        """Track execution of a specific line"""
        with self.lock:
            if module_name not in self.coverage_data:
                self.coverage_data[module_name] = set()
            self.coverage_data[module_name].add(line_no)

    def get_coverage_report(self, module_name: str = None) -> Dict:
        """Generate coverage report"""
        with self.lock:
            if module_name:
                executed = len(self.coverage_data.get(module_name, set()))
                return {
                    'module': module_name,
                    'lines_executed': executed,
                    'executed_lines': sorted(list(self.coverage_data.get(module_name, set())))
                }
            else:
                return {
                    'total_modules': len(self.coverage_data),
                    'modules': {name: len(lines) for name, lines in self.coverage_data.items()},
                    'coverage_data': {name: sorted(list(lines)) for name, lines in self.coverage_data.items()}
                }

    def save_report(self, output_path: str):
        """Save coverage report to file"""
        report = self.get_coverage_report()
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

# Global instance
_enterprise_coverage_tracker = EnterpriseCoverageTracker()

def instrument_module_for_coverage(module_path: str, module_name: str) -> None:
    """
    Instrument a Python module for coverage tracking using AST transformation

    Args:
        module_path: Path to the Python file
        module_name: Name of the module for tracking
    """
    try:
        # Read source code
        with open(module_path, 'r', encoding='utf-8') as f:
            source_code = f.read()

        # Parse AST
        tree = ast.parse(source_code, filename=module_path)

        # Instrument with coverage tracking
        instrumenter = EnterpriseCoverageInstrumenter(module_name)
        instrumented_tree = instrumenter.visit(tree)

        # Add the tracker import at the top
        tracker_import = ast.Import(
            names=[ast.alias(name='scripts.enterprise_coverage', asname=None)]
        )
        instrumented_tree.body.insert(0, tracker_import)

        # Convert back to source
        instrumented_code = compile(instrumented_tree, filename=module_path, mode='exec')

        # Execute in a controlled environment with our tracker
        module_globals = {
            '_enterprise_coverage_tracker': _enterprise_coverage_tracker,
            '__name__': module_name,
            '__file__': module_path,
        }

        # Add builtins
        module_globals.update(__builtins__ if isinstance(__builtins__, dict) else {'__builtins__': __builtins__})

        exec(instrumented_code, module_globals)

        print(f"✅ Successfully instrumented {module_name} for enterprise coverage tracking")

    except Exception as e:
        print(f"❌ Failed to instrument {module_name}: {e}")
        # Fallback: just execute normally without instrumentation
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print(f"⚠️ Executed {module_name} without instrumentation due to error")

def get_enterprise_coverage_report(output_path: str = None) -> Dict:
    """Get comprehensive enterprise coverage report"""
    tracker = EnterpriseCoverageTracker()

    if output_path:
        tracker.save_report(output_path)

    return tracker.get_coverage_report()

if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        module_path = sys.argv[1]
        module_name = sys.argv[2] if len(sys.argv) > 2 else Path(module_path).stem

        print(f"Instrumenting {module_name} from {module_path}")
        instrument_module_for_coverage(module_path, module_name)

        # Print coverage report
        import time
        time.sleep(1)  # Allow some execution
        report = get_enterprise_coverage_report()
        print(json.dumps(report, indent=2))
