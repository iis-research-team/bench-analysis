import csv
import re
import os
from typing import List, Optional


class DataProcessor:
    @staticmethod
    def is_valid_text(text: str) -> bool:
        if not text or not isinstance(text, str):
            return False
        if re.match(r'^\s*[\{\}\[\]"]+\s*$', text):
            return False
        if re.match(r'^\s*[a-z_]+:\s*[\{\}\[\]]\s*$', text):
            return False
        if text.count('"') > 10 or text.count('{') > 3 or text.count('}') > 3:
            return False
        return True

    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""
        
        text = re.sub(r'^"+|"+$', '', text.strip())
        text = re.sub(r'\\"', '"', text)
        text = re.sub(r'\s*\{\s*"[^"]+"\s*:\s*', '', text)
        text = re.sub(r'\s*\}\s*$', '', text)
        
        return text

    def process_csv(self, file_path: str, selected_fields: Optional[List[str]] = None) -> List[str]:
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
                print("We'll try to clean the data, but please check your original file")

            if selected_fields is None:
                print("\nAvailable fields in the CSV file:")
                for i, header in enumerate(cleaned_headers, 1):
                    print(f"{i}. {header}")
                
                user_input = input("\nEnter the numbers OR names of fields you want to extract (comma-separated): ").strip()
                selected_fields = [name.strip() for name in user_input.split(',')]

            if all(field.replace(',', '').replace(' ', '').isdigit() for field in selected_fields):
                selected_indices = [int(i.strip()) - 1 for i in selected_fields]
            else:
                selected_indices = [cleaned_headers.index(name) for name in selected_fields]

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
                    
                selected_values = []
                for idx in selected_indices:
                    if idx >= len(cleaned_row):
                        raise IndexError(f"Index {idx} is out of range for row {row_num}")
                    selected_values.append(cleaned_row[idx])
                
                concatenated = ' '.join(selected_values)
                if self.is_valid_text(concatenated):
                    result_strings.append(concatenated)
                    valid_rows += 1
                else:
                    invalid_rows += 1
                    print(f"Warning: Row {row_num} contained malformed JSON data and was skipped")

            print(f"\nProcessing complete. Results:")
            print(f"Valid rows processed: {valid_rows}")
            if invalid_rows > 0:
                print(f"Rows skipped due to malformed data: {invalid_rows}")
                print("Note: Some rows contained JSON artifacts or formatting issues")
                print("Please check your original file if this count seems high")
            
            return result_strings


if __name__ == "__main__":
    processor = DataProcessor()
    try:
        file_path = input("Enter the path to your CSV file: ")
        results = processor.process_csv(file_path)
        print("\nFirst 10 processed strings:")
        for s in results[:10]:
            print(s)
    except Exception as e:
        print(f"Error: {e}")