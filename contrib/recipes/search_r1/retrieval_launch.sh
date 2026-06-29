conda activate retriever

file_path=data
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

python scripts/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --search-batch-size 32 \
                                            --search-batch-wait-ms 10 \
                                            --faiss_gpu \
                                            --max-process-num 8
