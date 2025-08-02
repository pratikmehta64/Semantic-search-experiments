import sys, json
import numpy as np
def plot_comparison(submissions):
        import matplotlib.pyplot as plt
        
        def plot_single(submissions, ys, y_label, title, filename):
            if not submissions:
                print("No submissions to compare.")
                return
            query_names = list(submissions[0].keys())
            x = range(len(query_names))
            plt.figure(figsize=(12, 6))
            for i,y in enumerate(ys):
                print(f"Submission {i+1} - {y_label}:")
                print(np.median(y), np.mean(y))
                plt.bar([x_i + i*0.2 for x_i in x], y, width=0.2, label=y_label, align='center')
                
            plt.ylabel(y_label)
            plt.xlabel('Queries')
            
            
            plt.title(title)
            plt.xticks(x, query_names, rotation=45)
            plt.legend([f'Submission {i+1}' for i in range(len(submissions))])
            
            plt.tight_layout()
            plt.savefig('results/' + filename)
        
        avg_final_scores = [[sub['avg_final_score'] for sub in submission.values()] for submission in submissions]
        avg_soft_scores = [[sub['avg_soft_score'] for sub in submission.values()] for submission in submissions]
        avg_hard_scores = [[sub['avg_hard_score'] for sub in submission.values()] for submission in submissions]

        plot_single(submissions, ys=avg_final_scores, y_label='Average Final Score', title='Submission score Comparison', filename='Submissions_comparisions_avg_final_score.png')
        plot_single(submissions, ys=avg_soft_scores, y_label='Average Soft Score', title='Submission score Comparison', filename='Submissions_comparisions_avg_soft_score.png')
        plot_single(submissions, ys=avg_hard_scores, y_label='Average Hard Score', title='Submission score Comparison', filename='Submissions_comparisions_avg_hard_score.png')
        
if __name__ == "__main__":
    filenames = [arg for arg in sys.argv[1:]]   
    submissions = []
    for filename in filenames:
        submission = {}
        with open(filename, 'r') as file:
            submission_scores = json.load(file)
        
        for query in submission_scores['results']:
            
            query_name = submission_scores['results'][query]['config_name']
            query_avg_final_score = submission_scores['results'][query]['average_final_score']
            query_soft_scores = submission_scores['results'][query]['average_soft_scores']
            query_avg_soft_score = sum([score_details['average_score'] for score_details in query_soft_scores]) / len(query_soft_scores) if query_soft_scores else 0
            query_hard_scores = submission_scores['results'][query]['average_hard_scores']
            query_avg_hard_score = sum([score_details['pass_rate'] for score_details in query_hard_scores]) / len(query_hard_scores) if query_hard_scores else 0
            
            submission[query_name] = {
                'avg_final_score': query_avg_final_score,
                'avg_soft_score': query_avg_soft_score,
                'avg_hard_score': query_avg_hard_score,
                'soft_scores': query_soft_scores,
                'hard_scores': query_hard_scores
            }
        submissions.append(submission)
    plot_comparison(submissions)
    
    