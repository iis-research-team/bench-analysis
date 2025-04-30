import csv
import re
import os
from typing import List, Optional


class DataProcessor:
    """Processes CSV files that may contain JSON-formatted text data.
    
    Provides functionality to:
    - Clean text fields containing JSON artifacts
    - Validate text content
    - Select and extract specific fields
    - Process data while tracking valid/invalid rows
    """

    @staticmethod
    def is_valid_text(text: str) -> bool:
        """Check if text contains meaningful content (not empty or JSON structure).
        
        Args:
            text: String to validate
            
        Returns:
            bool: True if text is valid (non-empty and not JSON formatting)
        """
        if not text or not isinstance(text, str):
            return False
        if re.match(r'^\s*[\{\}\[\]"]+\s*$', text):
            return False
        return True

    @staticmethod
    def clean_text(text: Optional[str]) -> str:
        """Remove JSON artifacts and clean text formatting.
        
        Args:
            text: Input text that may contain JSON formatting (can be None)
            
        Returns:
            str: Cleaned text (empty string if input was None)
        """
        if text is None:
            return ""

        text = re.sub(r'^"+|"+$', '', text.strip())
        text = re.sub(r'\\"', '"', text)
        text = re.sub(r'\s*\{\s*"[^"]+"\s*:\s*', '', text)
        text = re.sub(r'\s*\}\s*$', '', text)

        return text

    def process_csv(
        self,
        file_path: str,
        selected_fields: Optional[List[str]] = None,
        display_results: bool = False
    ) -> List[str]:
        """Process CSV file and extract selected fields as clean text strings.
        
        Args:
            file_path: Path to CSV file to process
            selected_fields: List of field names/indices to extract (optional)
            display_results: Whether to print first 10 results (default: False)
            
        Returns:
            List[str]: Cleaned concatenated strings from selected fields
            
        Raises:
            FileNotFoundError: If specified file doesn't exist
            ValueError: For empty files or invalid headers
            IndexError: If field indices are out of range
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at {file_path}")

        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)

            try:
                headers = next(reader)
            except StopIteration:
                raise ValueError("File is empty or not properly formatted CSV")

            cleaned_headers = [self.clean_text(h) for h in headers]

            if any(not self.is_valid_text(h) for h in cleaned_headers):
                print("\nWarning: Some headers contain JSON artifacts")
                print("This suggests possible JSON-to-CSV conversion issues")

            if selected_fields is None:
                selected_fields = cleaned_headers

            selected_indices = self._parse_selected_fields(selected_fields, cleaned_headers)

            file.seek(0)
            next(reader)

            valid_rows = 0
            invalid_rows = 0
            result_strings = []

            for row_num, row in enumerate(reader, 1):
                cleaned_row = [self.clean_text(field) for field in row]
                if not all(self.is_valid_text(field) for field in cleaned_row):
                    invalid_rows += 1
                    continue

                try:
                    selected_values = [cleaned_row[idx] for idx in selected_indices]
                except IndexError as e:
                    raise IndexError(f"Field index out of range for row {row_num}") from e

                concatenated = ' '.join(selected_values)
                if self.is_valid_text(concatenated):
                    result_strings.append(concatenated)
                    valid_rows += 1
                else:
                    invalid_rows += 1

            self._print_processing_summary(valid_rows, invalid_rows)

            if display_results and result_strings:
                print("\nFirst 10 processed strings:")
                for s in result_strings[:10]:
                    print(s)

            return result_strings

    def _parse_selected_fields(
        self,
        selected_fields: List[str],
        cleaned_headers: List[str]
    ) -> List[int]:
        """Convert field names/indices to column indices.
        
        Args:
            selected_fields: List of field names or 1-based indices
            cleaned_headers: List of cleaned header names
            
        Returns:
            List[int]: 0-based column indices
            
        Raises:
            ValueError: If field names don't exist in headers
        """
        if all(field.replace(',', '').replace(' ', '').isdigit() for field in selected_fields):
            return [int(i.strip()) - 1 for i in selected_fields]
        else:
            try:
                return [cleaned_headers.index(name) for name in selected_fields]
            except ValueError as e:
                raise ValueError("One or more specified field names don't exist in CSV headers") from e

    def _print_processing_summary(self, valid_rows: int, invalid_rows: int) -> None:
        """Print processing statistics.
        
        Args:
            valid_rows: Count of successfully processed rows
            invalid_rows: Count of skipped rows due to formatting issues
        """
        print(f"\nProcessing complete. Results:")
        print(f"Valid rows processed: {valid_rows}")
        if invalid_rows > 0:
            print(f"Rows skipped due to malformed data: {invalid_rows}")
            print("Note: Some rows contained JSON artifacts or formatting issues")


if __name__ == "__main__":
    processor = DataProcessor()
    file_path = 'data.csv'
    selected_fields = ['1', '2']  
    
    try:
        results = processor.process_csv(
            file_path=file_path,
            selected_fields=selected_fields,
            display_results=True
        )
    except Exception as e:
        print(f"Error processing file: {e}")