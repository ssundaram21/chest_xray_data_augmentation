#!/bin/bash
#SBATCH -c 4
#SBATCH --array=2-3
#SBATCH --job-name=training
#SBATCH --mem=25GB
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=any-gpu
#SBATCH --partition=cbmm

#SBATCH -D ./log/

cd /local/nhulkund/UROP/6.819FinalProjectRAMP
export CUDA_VISIBLE_DEVICES=2

#list = {1..18}

task(){
      python /local/nhulkund/UROP/6.819FinalProjectRAMP/main.py --idx=$1 --user="neha" --datasetsplit=1
      python /local/nhulkund/UROP/6.819FinalProjectRAMP/main.py --idx=$1 --user="neha" --datasetsplit=10
      python /local/nhulkund/UROP/6.819FinalProjectRAMP/main.py --idx=$1 --user="neha" --datasetsplit=50
      python /local/nhulkund/UROP/6.819FinalProjectRAMP/main.py --idx=$1 --user="neha" --datasetsplit=100
}

#for int in {1..6}; do task "$int" & done
for int in {2..3}
do
    task "$int"
done

#parallel -j0 ::::  /local/nhulkund/UROP/6.819FinalProjectRAMP/main.py --idx={1..18} --user="neha"

