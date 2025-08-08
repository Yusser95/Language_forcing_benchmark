#!/bin/bash


languages=("bo" "mt" "it" "es" "de" "ja" )
# "ar" "zh" "af" "nl" "fr" "pt" "ru" "ko" "hi" "tr" "pl" "sv" "da" "no" "en"
model="meta-llama/Llama-3.1-8B" 
#"mistralai/Mistral-Nemo-Base-2407"
#"google/gemma-2-9b"


for lang in "${languages[@]}"
do
    echo "Running activations.py for language: $lang"
    python force_benchmark/activations/activations.py $model $lang
done

