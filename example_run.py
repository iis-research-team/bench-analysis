from benchmark_processor import mera
from text_analyzer import TextAnalyzer
from text_analysis_saver import save_analysis_to_file   


mera_data = mera("data/task (14).json", 
                extract_fields=['text'],
                text_only=True) 


analyzer = TextAnalyzer(mera_data)
results = analyzer.analyze()
save_analysis_to_file(results, dataset_name="ruEthics")
