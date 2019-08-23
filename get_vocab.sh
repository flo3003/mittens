sort -k1n glove-python/word_mapping.csv | awk -F ',' '{print $2}' |sed '1d' > vocabulary.csv
echo "word_a,word_b,cooccurrence" > coo_matrix.csv
awk -F ',' '{print $1-1","$2-1","$3}' glove-python/coo_matrix.csv | sed '1d' | sort -t ',' -k1n -k2n >> coo_matrix.csv
