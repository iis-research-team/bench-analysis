import csv
import re

class data_processor:
    def __init__(self):
        pass

    def is_valid_text(self, text):
        if not text or not isinstance(text, str):
            return False
        if re.match(r'^\s*[\{\}\[\]"]+\s*$', text):
            return False
        if re.match(r'^\s*[a-z_]+:\s*[\{\}\[\]]\s*$', text):
            return False
        if text.count('"') > 10 or text.count('{') > 3 or text.count('}') > 3:
            return False
        return True

    def clean_text(self, text):
        if not text:
            return ""
        
        text = re.sub(r'^"+|"+$', '', text.strip())
        text = re.sub(r'\\"', '"', text)
        
        text = re.sub(r'\s*\{\s*"[^"]+"\s*:\s*', '', text)
        text = re.sub(r'\s*\}\s*$', '', text)
        
        return text

    def process_csv(self, file_path=None):
        if not file_path:
            file_path = input("Enter the path to your CSV file: ")
        
        try:
            with open(file_path, mode='r', encoding='utf-8') as file:
                try:
                    reader = csv.reader(file)
                    headers = next(reader)
                    
                    cleaned_headers = [self.clean_text(h) for h in headers]
                    if any(not self.is_valid_text(h) for h in cleaned_headers):
                        print("\nWarning: Some headers appear to contain JSON artifacts")
                        print("This suggests possible conversion issues from JSON to CSV")
                        print("We'll try to clean the data, but please check your original file")
                    
                    print("\nAvailable fields in the CSV file:")
                    for i, header in enumerate(cleaned_headers, 1):
                        print(f"{i}. {header}")
                    
                    user_input = input("\nEnter the numbers OR names of fields you want to extract (comma-separated): ").strip()
                    
                    if user_input.replace(',', '').replace(' ', '').isdigit():
                        selected_indices = [int(i.strip()) - 1 for i in user_input.split(',')]
                    else:
                        selected_names = [name.strip() for name in user_input.split(',')]
                        selected_indices = [cleaned_headers.index(name) for name in selected_names]
                    
                    file.seek(0)
                    next(reader)
                    
                    valid_rows = 0
                    invalid_rows = 0
                    result_strings = []
                    
                    for row_num, row in enumerate(reader, 1):
                        try:
                            cleaned_row = [self.clean_text(field) for field in row]
                            if not all(self.is_valid_text(field) for field in cleaned_row):
                                invalid_rows += 1
                                continue
                                
                            selected_values = []
                            for idx in selected_indices:
                                if idx < len(cleaned_row):
                                    selected_values.append(cleaned_row[idx])
                            
                            concatenated = ' '.join(selected_values)
                            if self.is_valid_text(concatenated):
                                result_strings.append(concatenated)
                                valid_rows += 1
                            else:
                                invalid_rows += 1
                                print(f"Warning: Row {row_num} contained malformed JSON data and was skipped")
                        except Exception as e:
                            invalid_rows += 1
                            print(f"Warning: Row {row_num} had formatting issues and was skipped")
                            continue
                    
                    print(f"\nProcessing complete. Results:")
                    print(f"Valid rows processed: {valid_rows}")
                    if invalid_rows > 0:
                        print(f"Rows skipped due to malformed data: {invalid_rows}")
                        print("Note: Some rows contained JSON artifacts or formatting issues")
                        print("Please check your original file if this count seems high")
                    
                    print("\nResulting strings:")
                    for s in result_strings:
                        print(s)
                    
                    return result_strings
                    
                except StopIteration:
                    raise ValueError("File appears empty or not properly formatted as CSV")
        
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
        except ValueError as e:
            print(f"Data Validation Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    processor = data_processor()
    processor.process_csv()