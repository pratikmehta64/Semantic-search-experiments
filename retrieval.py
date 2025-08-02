import requests
import turbopuffer
import voyageai
import sys, os
import json
import matplotlib.pyplot as plt
from prompts import prompt_1
from init import (
                OPENAI_API_KEY,
                TURBOPUFFER_API_KEY, 
                VOYAGEAI_API_KEY, 
                MY_EMAIL,
                EVAL_ENDPOINT,
                GRADE_ENDPOINT
                )

VO = voyageai.Client(
    api_key=VOYAGEAI_API_KEY
    )

TPUF = turbopuffer.Turbopuffer(
    api_key=TURBOPUFFER_API_KEY,
    region="aws-us-west-2",
)

TPUF_NAMESPACE_NAME = "pratik_mehta_tpuf_key"
NS = TPUF.namespace(TPUF_NAMESPACE_NAME)
QUERY_CHAR_LIMIT = 1024
class Retrieval:
    def __init__(self):
        pass
    
    def fetch_queries(self, filename: str):
        with open(filename, 'r') as file:
            queries = json.load(file)
        return queries
    
    def construct_query(self, query: str):
        full_query = f"""{prompt_1} 
                        [Title]
                        {query['Title']} \n
                        [Hard Criteria]\n
                        {query['Hard Criteria']} \n
                        [Soft Criteria]\n
                        {query['Soft Criteria']}
                        [Description]\n
                        {query['Natural Language Description']} \n
                        """
        full_query = full_query[:QUERY_CHAR_LIMIT]  # Ensure the query does not exceed the character limit
        
        return full_query
    
    def reciprocal_rank_fusion(self, result_lists, k = 60): # simple way to fuse results based on position
        
        scores = {} 
        all_results = {} 
        for results in result_lists:
            for rank, item in enumerate(results, start=1):
                scores[item.id] = scores.get(item.id, 0) + 1.0 / (k + rank)
                all_results[item.id] = item
        return [
            setattr(all_results[doc_id], '$dist', score) or all_results[doc_id]
            for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ][:10]  # Return top 10 results
          
    def retrieve_results(self, all_queries):
        try:
            all_vector_results = []
            all_fts_results = []
            all_fusion_results = []
            for query in all_queries:
                constructed_query = self.construct_query(query)
                query_embedding = VO.embed([constructed_query], model="voyage-3", input_type="query")
                response = NS.multi_query(
                    queries=[
                        {
                            "rank_by": ("vector", "ANN", query_embedding.embeddings[0]),
                            "top_k": 10,
                            "include_attributes": True,
                        },
                        {
                            "rank_by": ("rerank_summary", "BM25", constructed_query),
                            "top_k": 10,
                            "include_attributes": True,
                        },
                    ]
                )
                vector_result, fts_result = response.results[0].rows, response.results[1].rows
                fused_result = self.reciprocal_rank_fusion([vector_result, fts_result])
                
                all_vector_results.append(vector_result)  
                all_fts_results.append(fts_result)
                all_fusion_results.append(fused_result)     
            return [all_vector_results, all_fts_results, all_fusion_results]
        
        except Exception as e:
            print(f"Error processing query: {e}")
    
    def evaluate(self, all_queries, all_results, results_path=""):
        vector_results, fts_results, fused_results = all_results
        all_scores = []
        
        vector_result_filename = results_path + "vector_retrieval_results.json"
        fts_result_filename = results_path + "fts_retrieval_results.json"
        fused_results_filename = results_path + "fused_retrieval_results.json"
        
        with open(vector_result_filename, "w") as file:
            pass
        with open(fts_result_filename, "w") as file:
            pass
        with open(fused_results_filename, "w") as file:
            pass
        
        def fetch_scores_online(ids: list, query: dict):
            URL = EVAL_ENDPOINT
            response = requests.post(
                URL,
                json={
                    "config_path": query["Yaml File"],
                    "object_ids": ids
                },
                headers={"Content-Type": "application/json",
                         "Authorization": MY_EMAIL}
            )
            return response
        
        for i, query in enumerate(all_queries):
            vector_ids = [result.id for result in vector_results[i]]
            fts_ids = [result.id for result in fts_results[i]]
            fused_ids = [result.id for result in fused_results[i]]
            
            vector_response = fetch_scores_online(vector_ids, query)
            print(f"vector_response secured")
            
            with open(vector_result_filename, "a+") as file:
                vector_response = vector_response.json()
                file.write(f"{vector_response}")
            print(f"vector response written to file")
            
            fts_response = fetch_scores_online(fts_ids, query)
            print(f"fts_response secured")        
            
            with open(fts_result_filename, "a+") as file:
                fts_response = fts_response.json()
                file.write(f'{fts_response}') 
            print(f"fts response written to file")
            
            fused_response = fetch_scores_online(fused_ids, query)
            print(f"fused_response secured")
            
            with open(fused_results_filename, "a+") as file:
                fused_response = fused_response.json()
                file.write(f'{fused_response}')
            print(f"fused response written to file")
            
            vector_avg_final_search_score = vector_response.get("average_final_score")
            vector_soft_criteria_scores = vector_response.get("average_soft_scores")
            vector_hard_criteria_scores = vector_response.get("average_hard_scores")
            
            fts_avg_final_search_score = fts_response.get("average_final_score")
            fts_soft_criteria_scores = fts_response.get("average_soft_scores")
            fts_hard_criteria_scores = fts_response.get("average_hard_scores")
            
            fusion_avg_final_search_score = fused_response.get("average_final_score")
            fusion_soft_criteria_scores = fused_response.get("average_soft_scores")
            fusion_hard_criteria_scores = fused_response.get("average_hard_scores")

            query_scores = {
                "query": query["Title"],
                "vector_avg_final_search_score": vector_avg_final_search_score,
                "vector_soft_criteria_scores": vector_soft_criteria_scores,
                "vector_hard_criteria_scores": vector_hard_criteria_scores,
                "fts_avg_final_search_score": fts_avg_final_search_score,
                "fts_soft_criteria_scores": fts_soft_criteria_scores,
                "fts_hard_criteria_scores": fts_hard_criteria_scores,
                "fusion_avg_final_search_score": fusion_avg_final_search_score,
                "fusion_soft_criteria_scores": fusion_soft_criteria_scores,
                "fusion_hard_criteria_scores": fusion_hard_criteria_scores
            }
            all_scores.append(query_scores)
        
        print("Evaluation completed for all queries.")
        return all_scores   
        
    
    def submit(self, all_queries, all_results):
        vector_results, fts_results, fused_results = all_results
        config_candidates = {}
        URL = GRADE_ENDPOINT
        for i, query in enumerate(all_queries):
            config_candidates[query["Yaml File"]] = [result.id for result in fused_results[i]]
        response = requests.post(
            URL,
            json={"config_candidates": config_candidates},
            headers={
                     "Content-Type": "application/json",
                     "Authorization": MY_EMAIL
            }
        )
        print(f"Vector submission response: {response.json()}")

def weird_division(n, d):
    return n / d if d else 0

def plot_scores(all_queries, results_root_dir='./results/'):
    cross_experiment_vector_avg_final_search_scores = []
    cross_experiment_vector_avg_hard_criteria_scores = []
    cross_experiment_vector_avg_soft_criteria_scores = []
    
    cross_experiment_fts_avg_final_search_scores = []
    cross_experiment_fts_avg_hard_criteria_scores = []
    cross_experiment_fts_avg_soft_criteria_scores = []
    
    cross_experiment_fusion_avg_final_search_scores = []
    cross_experiment_fusion_avg_hard_criteria_scores = []
    cross_experiment_fusion_avg_soft_criteria_scores = []
    
    for dirname in os.listdir(results_root_dir):
        full_path = os.path.join(results_root_dir, dirname)
        if os.path.isdir(full_path):
            scores_file = os.path.join(full_path, 'eval_online_scores.json')
        else:
            continue
        with open(scores_file, 'r') as file:
            scores = json.load(file)[0]
            
            fts_avg_final_search_scores = []
            fts_avg_hard_criteria_scores = []
            fts_avg_soft_criteria_scores = []
            
            vector_avg_final_search_scores = []
            vector_avg_hard_criteria_scores = []
            vector_avg_soft_criteria_scores = []
            
            fusion_avg_final_search_scores = []
            fusion_avg_hard_criteria_scores= []
            fusion_avg_soft_criteria_scores = []
            
            for _ in all_queries:
                num_zeros = 0
                vector_avg_final_search_score = scores.get("vector_avg_final_search_score")
                fts_avg_final_search_score = scores.get("fts_avg_final_search_score")
                fusion_avg_final_search_score = scores.get("fusion_avg_final_search_score")
                
                fts_avg_final_search_scores.append(fts_avg_final_search_score)
                
                vector_avg_final_search_scores.append(vector_avg_final_search_score)
                
                fusion_avg_final_search_scores.append(fusion_avg_final_search_score)
                
                vector_soft_criteria_scores = scores.get("vector_soft_criteria_scores")
                
                fts_soft_criteria_scores = scores.get("fts_soft_criteria_scores")
                
                fusion_soft_criteria_scores = scores.get("fusion_soft_criteria_scores")
                
                vector_hard_criteria_scores = scores.get("vector_hard_criteria_scores")
                
                fts_hard_criteria_scores = scores.get("fts_hard_criteria_scores")
                
                fusion_hard_criteria_scores = scores.get("fusion_hard_criteria_scores")
                
                vector_avg_soft_criteria_scores.append(weird_division(sum([obj['average_score'] for obj in vector_soft_criteria_scores]), len(vector_soft_criteria_scores)))
                fts_avg_soft_criteria_scores.append(weird_division(sum([obj['average_score'] for obj in fts_soft_criteria_scores]), len(fts_soft_criteria_scores)))
                fusion_avg_soft_criteria_scores.append(weird_division(sum([obj['average_score'] for obj in fusion_soft_criteria_scores]), len(fusion_soft_criteria_scores)-num_zeros))
                
                vector_avg_hard_criteria_scores.append(weird_division(sum([obj['pass_rate'] for obj in vector_hard_criteria_scores]), len(vector_hard_criteria_scores)))
                fts_avg_hard_criteria_scores.append(weird_division(sum([obj['pass_rate'] for obj in fts_hard_criteria_scores]), len(fts_hard_criteria_scores)))
                fusion_avg_hard_criteria_scores.append(weird_division(sum([obj['pass_rate'] for obj in fusion_hard_criteria_scores]), len(fusion_hard_criteria_scores)-num_zeros))
                
        cross_experiment_vector_avg_final_search_scores.append(weird_division(sum(vector_avg_final_search_scores),len(vector_avg_final_search_scores)))
        cross_experiment_vector_avg_hard_criteria_scores.append(weird_division(sum(vector_avg_hard_criteria_scores),len(vector_avg_hard_criteria_scores)))
        cross_experiment_vector_avg_soft_criteria_scores.append(weird_division(sum(vector_avg_soft_criteria_scores),len(vector_avg_soft_criteria_scores)))
        cross_experiment_fts_avg_final_search_scores.append(weird_division(sum(fts_avg_final_search_scores),len(fts_avg_final_search_scores)))
        cross_experiment_fts_avg_hard_criteria_scores.append(weird_division(sum(fts_avg_hard_criteria_scores),len(fts_avg_hard_criteria_scores)))
        cross_experiment_fts_avg_soft_criteria_scores.append(weird_division(sum(fts_avg_soft_criteria_scores),len(fts_avg_soft_criteria_scores)))
        cross_experiment_fusion_avg_final_search_scores.append(weird_division(sum(fusion_avg_final_search_scores),len(fusion_avg_final_search_scores)))
        cross_experiment_fusion_avg_hard_criteria_scores.append(weird_division(sum(fusion_avg_hard_criteria_scores),len(fusion_avg_hard_criteria_scores)))
        cross_experiment_fusion_avg_soft_criteria_scores.append(weird_division(sum(fusion_avg_soft_criteria_scores),len(fusion_avg_soft_criteria_scores)))
    
    fig, axs = plt.subplots(3, 1)
    
    X = range(len(cross_experiment_vector_avg_final_search_scores))
    y1 = cross_experiment_vector_avg_final_search_scores
    y2 = cross_experiment_fts_avg_final_search_scores
    y3 = cross_experiment_fusion_avg_final_search_scores
    axs[0].plot(X, y1, label='Avg Vector Search Final Score')
    axs[0].plot(X, y2, label='Avg FTS Search Final Score')
    axs[0].plot(X, y3, label='Avg Fusion Search Final Score')
    axs[0].legend()
    
    for ax in axs:
        ax.set(xlabel='Experiment Index', ylabel='Score')
        plt.xticks(X)
    
    y4 = cross_experiment_vector_avg_hard_criteria_scores
    y5 = cross_experiment_fts_avg_hard_criteria_scores
    y6 = cross_experiment_fusion_avg_hard_criteria_scores
    axs[1].plot(X, y4, label='Avg Vector Hard Criteria Score')
    axs[1].plot(X, y5, label='Avg FTS Hard Criteria Score')
    axs[1].plot(X, y6, label='Avg Fusion Hard Criteria Score')
    axs[1].legend()
    
    for ax in axs:
        ax.set(xlabel='Experiment Index', ylabel='Score')
        plt.xticks(X)
    
    y7 = cross_experiment_vector_avg_soft_criteria_scores
    y8 = cross_experiment_fts_avg_soft_criteria_scores
    y9 = cross_experiment_fusion_avg_soft_criteria_scores
    axs[2].plot(X, y7, label='Avg Vector Soft Criteria Score')
    axs[2].plot(X, y8, label='Avg FTS Soft Criteria Score')
    axs[2].plot(X, y9, label='Avg Fusion Soft Criteria Score')
    axs[2].legend()
    
    for ax in axs:
        ax.set(xlabel='Experiment Index', ylabel='Score')
        plt.xticks(X)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_root_dir, 'cross_experiment_scores.png'))

if __name__ == "__main__":  
    retriever = Retrieval()
    all_queries = retriever.fetch_queries("queries.json")
    all_results = retriever.retrieve_results(all_queries)

    new_results_path = sys.argv[1] if len(sys.argv) > 1 else ""
    os.makedirs(new_results_path, exist_ok=True)
    eval_online_scores = retriever.evaluate(all_queries, all_results, new_results_path)

    with open(new_results_path + "eval_online_scores.json", "w") as file:
        json.dump(eval_online_scores, file, indent=4)

    plot_scores(all_queries)

    retriever.submit(all_queries, all_results)