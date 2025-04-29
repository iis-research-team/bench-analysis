import csv
import re
import os
from typing import List, Optional


class DataProcessor:
    @staticmethod
    def is_valid_text(text: str) -> bool:
        if not text or not isinstance(text, str):
            return False
        # this checks for patterns that indicate malformed JSON or empty structures
        if re.match(r'^\s*[\{\}\[\]"]+\s*$', text):
            return False
        return True

    @staticmethod
    def clean_text(text: Optional[str]) -> str:
        if text is None:
            return ""

        text = re.sub(r'^"+|"+$', '', text.strip())
        text = re.sub(r'\\"', '"', text)
        text = re.sub(r'\s*\{\s*"[^"]+"\s*:\s*', '', text)
        text = re.sub(r'\s*\}\s*$', '', text)

        return text

    def process_csv(self,
                    file_path: str,
                    selected_fields: Optional[List[str]] = None,
                    display_results: bool = False,
                    user_input: Optional[str] = None) -> List[str]:

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at {file_path}")

        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)

            try:
                headers = next(reader)
            except StopIteration:
                raise ValueError("File appears empty or not properly formatted as CSV")

            cleaned_headers = [self.clean_text(h) for h in headers]

            if any(not self.is_valid_text(h) for h in cleaned_headers):
                print("\nWarning: Some headers appear to contain JSON artifacts")
                print("This suggests possible conversion issues from JSON to CSV")

            if selected_fields is None:
                selected_fields = self._get_user_selected_fields(cleaned_headers, user_input)

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

    def _get_user_selected_fields(
        self,
        cleaned_headers: List[str],
        user_input: Optional[str] = None
    ) -> List[str]:

        if user_input is None:
            print("\nAvailable fields in the CSV file:")
            for i, header in enumerate(cleaned_headers, 1):
                print(f"{i}. {header}")
            user_input = input("\nEnter the numbers OR names of fields you want to extract (comma-separated): ").strip()

        return [name.strip() for name in user_input.split(',') if name.strip()]

    def _parse_selected_fields(self,
                               selected_fields: List[str],
                               cleaned_headers: List[str]) -> List[int]:

        if all(field.replace(',', '').replace(' ', '').isdigit() for field in selected_fields):
            return [int(i.strip()) - 1 for i in selected_fields]
        else:
            try:
                return [cleaned_headers.index(name) for name in selected_fields]
            except ValueError as e:
                raise ValueError("One or more specified field names don't exist in the CSV headers") from e

    def _print_processing_summary(self, valid_rows: int, invalid_rows: int) -> None:
        print(f"\nProcessing complete. Results:")
        print(f"Valid rows processed: {valid_rows}")
        if invalid_rows > 0:
            print(f"Rows skipped due to malformed data: {invalid_rows}")
            print("Note: Some rows contained JSON artifacts or formatting issues")


if __name__ == "__main__":
    processor = DataProcessor()
    try:
        file_path = input("Enter the path to your CSV file: ").strip()
        user_input = input("Enter field numbers or names (comma-separated)").strip() or None
        results = processor.process_csv(file_path, display_results=True, user_input=user_input)
    except Exception as e:
        print(f"Error: {e}")
