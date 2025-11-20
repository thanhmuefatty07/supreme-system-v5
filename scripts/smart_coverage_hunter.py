#!/usr/bin/env python3
"""
Smart Coverage Hunter - Sequential AI Test Generation with Quota Management

Features:
- Sequential processing (no parallel spam)
- Smart API key rotation on quota exceeded
- Auto-sleep when all keys exhausted
- Real-time test validation
- Only keeps passing tests
"""

import os
import time
import subprocess
import logging
from pathlib import Path
import google.generativeai as genai
from google.api_core import exceptions
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger()

load_dotenv()

# Load all available API keys
API_KEYS = [os.getenv(f"GEMINI_API_KEY_{i}") for i in range(1, 7)]
VALID_KEYS = [k for k in API_KEYS if k]

if not VALID_KEYS:
    print("❌ Không tìm thấy API Keys.")
    exit()

current_key_index = 0
quota_exceeded_count = 0

def get_next_model():
    """Lấy model từ key tiếp theo, tự động xoay vòng"""
    global current_key_index
    key = VALID_KEYS[current_key_index]
    genai.configure(api_key=key)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    return model

def switch_key():
    """Chuyển sang key kế tiếp khi gặp lỗi Quota"""
    global current_key_index, quota_exceeded_count
    current_key_index = (current_key_index + 1) % len(VALID_KEYS)
    quota_exceeded_count += 1
    logger.warning(f"⚠️ Quota Exceeded. Switching to Key #{current_key_index + 1}... ({quota_exceeded_count} switches)")

def generate_with_retry(prompt):
    """Thử generate, nếu lỗi quota thì đổi key và thử lại"""
    max_global_retries = len(VALID_KEYS) * 2  # Thử 2 vòng toàn bộ keys

    for attempt in range(max_global_retries):
        try:
            model = get_next_model()
            # Thêm delay nhẹ để tránh spam
            time.sleep(2)
            response = model.generate_content(prompt)
            return response

        except exceptions.ResourceExhausted:
            switch_key()
            time.sleep(1)  # Nghỉ 1s khi đổi key

        except Exception as e:
            logger.error(f"❌ API Error: {str(e)}")
            return None

    # Nếu thử hết tất cả keys mà vẫn lỗi -> Ngủ dài
    logger.error("💤 All keys exhausted. Sleeping 60s...")
    time.sleep(60)
    return generate_with_retry(prompt)  # Đệ quy thử lại

def process_file(file_path):
    """Process a single file to generate tests"""
    global quota_exceeded_count

    module_name = Path(file_path).stem
    test_file = Path(f"tests/unit/test_{module_name}_smart.py")

    if test_file.exists():
        return  # Skip if already exists

    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()

    if len(code) < 50:  # Skip too small files
        return

    logger.info(f"🎯 Processing: {module_name}...")

    prompt = f"""
    Write a pytest file for `{module_name}`.
    Source code:

    ```
    {code}
    ```

    Requirements:
    - Use unittest.mock for ALL external dependencies
    - NO sys.modules mocking
    - Create realistic fixtures and mocks
    - Include error handling tests
    - Output ONLY python code block (no markdown)
    - Make tests actually runnable
    """

    response = generate_with_retry(prompt)
    if not response:
        return

    try:
        # Extract code from response
        test_code = response.text
        if '```python' in test_code:
            test_code = test_code.split('```python')[1].split('```')[0]
        elif '```' in test_code:
            test_code = test_code.split('```')[1].split('```')[0]

        test_code = test_code.strip()

        # Write test file
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_code)

        # Verify test immediately
        result = subprocess.run(
            ["pytest", str(test_file)],
            capture_output=True,
            text=True,
            timeout=15
        )

        if result.returncode == 0:
            logger.info(f"✅ KEEP: {module_name} (Test Passed)")
        else:
            os.remove(test_file)
            logger.info(f"🗑️ DELETE: {module_name} (Test Failed)")

    except Exception as e:
        logger.error(f"❌ Processing error for {module_name}: {e}")
        if test_file.exists():
            os.remove(test_file)

def main():
    """Main execution function"""
    # Find all Python files to process
    files = []
    for root, _, filenames in os.walk("src"):
        for f in filenames:
            if f.endswith(".py") and f != "__init__.py":
                files.append(os.path.join(root, f))

    logger.info(f"🚀 Smart Hunting: {len(files)} files with {len(VALID_KEYS)} keys.")
    logger.info("Strategy: Sequential processing with smart quota management")

    successful_tests = 0
    total_processed = 0

    # Process files sequentially (not parallel)
    for f in files:
        try:
            process_file(f)
            total_processed += 1

            # Progress update every 10 files
            if total_processed % 10 == 0:
                logger.info(f"📊 Progress: {total_processed}/{len(files)} files processed")

        except KeyboardInterrupt:
            logger.info("⏹️ Interrupted by user")
            break
        except Exception as e:
            logger.error(f"❌ Unexpected error processing {f}: {e}")
            continue

    logger.info(f"\n🏁 FINISHED. Processed {total_processed} files.")
    logger.info(f"📈 Quota exceeded events: {quota_exceeded_count}")
    logger.info("Check coverage with: pytest tests/unit/test_*_smart.py --cov=src")

if __name__ == "__main__":
    main()
