# Step 1: Vanilla LLM-based Skill Extraction
python3 summarize.py --input_file data/stackoverflow-data-science.jsonl --output_file data/steps.jsonl --log_file data/summarize.log --model qwen3-32b --retry 4 --process_num 10 --debug 0

for i in 1 2 3 4 5
do
  echo "Running iteration $i"
  # Step 2: Embedding-based Skill Clustering
  # Step 2.1: Skill Embedding
  python3 embed.py --layer 1 --model Qwen3-Embedding-4B
  
  # Step 2.2: Skill Clustering
  python3 cluster_raptor.py --layer 1
  
  # Step 3: LLM-based Skill Cluster Refinement
  # Step 3.1: Parent Skill-Aligned Skill Cluster Split
  python3 cluster.py --layer 1 --model qwen3-32b --retry 4 --process_num 10 --debug 0
  
  # Step 3.2: Redundant Skill Merge
  python3 cluster_dbscan.py --layer 1 --model Qwen3-Embedding-4B
  
  # Step 3.3: Parent Skill Summarization
  python3 formulate_cluster.py --layer 1
done

# Step 4: Entangled Skill Elimination
python3 remove_name_covers.py --layer 6

# (Optional) Detailed Skill Description Generation
python3 describe_skill.py --output_file data/skill-descriptions.jsonl --log_file data/describe_skill.log --model qwen3-32b --retry 4 --process_num 10 --debug 0