import os
from geneformer import TranscriptomeTokenizer

nproc = min(8, os.cpu_count() or 1)
tk = TranscriptomeTokenizer(custom_attr_name_dict={}, nproc=nproc)

print("Tokenizer start!!")
tk.tokenize_data(
    data_directory="./data/tutorial_h5ad/",
    output_directory="./data/tokenized/",
    output_prefix="tutorial_mouse",
    file_format="h5ad",
)
print("Tokenizer finished!!")