data_dir=/data/cdq/pet_data/data

cd ${data_dir}

# Yelp Reviews
mkdir yelp
cd yelp
wget https://s3.amazonaws.com/fast-ai-nlp/yelp_review_full_csv.tgz
tar -zxvf yelp_review_full_csv.tgz
mv yelp_review_full_csv/* .
rm -rf yelp*
cd ..

# Yahoo Questions
mkdir yahoo
cd yahoo
wget https://s3.amazonaws.com/fast-ai-nlp/yahoo_answers_csv.tgz
tar -zxvf yahoo_answers_csv.tgz
mv yahoo_answers_csv/* .
rm -rf yahoo*
cd ..

# MNLI
wget https://dl.fbaipublicfiles.com/glue/data/MNLI.zip
unzip MNLI.zip
rm -rf MNLI.zip

# SuperGLUE
wget https://dl.fbaipublicfiles.com/glue/superglue/data/v2/combined.zip
unzip combined.zip
rm combined.zip