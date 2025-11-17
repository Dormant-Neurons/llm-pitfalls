"""
Analysis and visualization of pitfall review data from CSV.

SETUP INSTRUCTIONS:
1. Create the data directory if it doesn't exist:
   mkdir -p data/
   
2. Download the Google Sheets data as CSV:
   - Open the Google Sheets document
   - Go to File > Download > Comma-separated values (.csv)
   - Save the file as 'pitfalls.csv' in the 'data/' directory
   
3. Run the analysis:
   python main.py
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_hex
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

sns.set_theme(style="whitegrid")

def calculate_fleiss_kappa_for_pitfall(ratings_matrix):
    """Calculate Fleiss' Kappa for inter-rater reliability using statsmodels."""
    ratings_table, _ = aggregate_raters(np.array(ratings_matrix))
    return fleiss_kappa(ratings_table, method='fleiss')

def fix_row_index_offset(hidden_rows):
    """Adjust hidden row indices to match DataFrame index after skipping headers."""
    return [row - 3 for row in hidden_rows]

def load_and_clean_data(csv_path, hidden_rows):
    """Load CSV, remove hidden rows, trim to last valid row, and extract pitfall columns."""
    df = pd.read_csv(csv_path, skiprows=1).replace({np.nan: None})
    df = df[~df.index.isin(fix_row_index_offset(hidden_rows))].reset_index(drop=True)

    # Keep only up to the last row where the second column is not empty
    last_row_index = df[df.iloc[:, 1].notna()].index[-1]  # type: ignore
    df = df.iloc[:last_row_index + 1] # type: ignore

    # Drop columns after the first empty column
    last_column = df.columns[df.columns.str.startswith("Unnamed")][0]
    last_column_index = df.columns.get_loc(last_column)
    df = df.iloc[:, :last_column_index]

    # Extract pitfall names from header
    with open(csv_path, 'r') as file:
        header = file.readline().split(',')
    pitfalls = [p.strip() for p in header[:last_column_index] if re.match(r'^P\d+\.\s', p.strip())]

    return df, pitfalls

def aggregate_stats(df, pitfalls):
    """Aggregate all statistics from the DataFrame into a single dictionary."""
    papers = []
    for i in range(0, len(df), 4):
        paper_pitfalls = {}
        for pitfall_idx, pitfall_name in enumerate(pitfalls):
            pitfall_column = "Pitfall applies?" if pitfall_idx == 0 else f"Pitfall applies?.{pitfall_idx}"
            paper_pitfalls[pitfall_name] = {
                "a": df.iloc[i][pitfall_column],
                "b": df.iloc[i + 1][pitfall_column],
                "final": df.iloc[i + 2][pitfall_column],
                "final_final": df.iloc[i + 3][pitfall_column],
            }
        papers.append({
            "title": df.iloc[i]["Paper Title"],
            "topic": df.iloc[i]["Topic"],
            "reviewer_a": df.iloc[i]["A"],
            "reviewer_b": df.iloc[i]["B"],
            "pitfalls": paper_pitfalls,
        })

    pitfall_stats = defaultdict(Counter)
    reviewer_agreement = Counter({'agree': 0, 'disagree': 0})
    topic_counts = Counter()
    pitfall_disagreement = Counter()
    pitfall_ratings = defaultdict(list)
    
    for paper in papers:
        topic_counts[paper['topic']] += 1
        for pitfall, vals in paper['pitfalls'].items():
            if vals.get('final_final') is not None:
                pitfall_stats[pitfall][vals['final_final']] += 1
            
            if vals['a'] == vals['b']:
                reviewer_agreement['agree'] += 1
            else:
                reviewer_agreement['disagree'] += 1
                pitfall_disagreement[pitfall] += 1
            
            if vals['a'] is not None and vals['b'] is not None:
                pitfall_ratings[pitfall].append([vals['a'], vals['b']])
    
    # Calculate Fleiss' Kappa for each pitfall and overall
    pitfall_kappa = {pitfall: calculate_fleiss_kappa_for_pitfall(ratings) 
                     for pitfall, ratings in pitfall_ratings.items() if ratings}
    
    all_ratings = [rating for ratings in pitfall_ratings.values() for rating in ratings]
    overall_kappa = calculate_fleiss_kappa_for_pitfall(all_ratings) if all_ratings else 0.0
    
    return {
        'pitfall_stats': dict((k, dict(v)) for k, v in pitfall_stats.items()),
        'reviewer_agreement': dict(reviewer_agreement),
        'topic_counts': dict(topic_counts),
        'pitfall_disagreement': dict(pitfall_disagreement),
        'pitfall_kappa': pitfall_kappa,
        'overall_kappa': overall_kappa,
    }

def compute_and_save_pitfall_percentages_per_topic(df, pitfalls, output_dir):
    """
    Computes the percentage of each final pitfall label per topic and saves it as JSON.
    Output format: { topic: { pitfall_name: { label: percentage, ... }, ... }, ... }
    """
    result = defaultdict(lambda: defaultdict(Counter))
    
    for i in range(0, len(df), 4):
        topic = df.iloc[i]["Topic"]
        for pitfall_idx, pitfall_name in enumerate(pitfalls):
            pitfall_column = "Pitfall applies?" if pitfall_idx == 0 else f"Pitfall applies?.{pitfall_idx}"
            final_final_label = df.iloc[i + 3][pitfall_column]
            if final_final_label is not None:
                result[topic][pitfall_name][final_final_label] += 1

    # Normalize to percentages
    percentages = {}
    for topic, pitfall_data in result.items():
        percentages[topic] = {}
        for pitfall, counts in pitfall_data.items():
            total = sum(counts.values())
            if total > 0:
                percentages[topic][pitfall] = {
                    label: round(count / total * 100, 2) for label, count in counts.items()
                }

    with open(output_dir / 'pitfall_percentages_per_topic.json', 'w') as f:
        json.dump(percentages, f, indent=2, sort_keys=True)
        
def compute_and_save_present_percentages_per_topic(df, pitfalls, output_dir):
    """
    Computes the percentage of 'Present' or 'Present (but discussed)' per topic, averaged over all pitfalls.
    Saves output as: pitfall_percentages_per_topic_only_present.json
    """
    present_labels = {'Present', 'Present (but discussed)'}
    topic_totals = defaultdict(int)
    topic_present = defaultdict(int)

    for i in range(0, len(df), 4):
        topic = df.iloc[i]["Topic"]
        topic_totals[topic] += len(pitfalls)
        for pitfall_idx in range(len(pitfalls)):
            pitfall_column = "Pitfall applies?" if pitfall_idx == 0 else f"Pitfall applies?.{pitfall_idx}"
            label = df.iloc[i + 3][pitfall_column]
            if label in present_labels:
                topic_present[topic] += 1

    percentages = {
        topic: round((topic_present[topic] / topic_totals[topic]) * 100, 2)
        for topic in topic_totals
    }

    with open(output_dir / 'pitfall_percentages_per_topic_only_present.json', 'w') as f:
        json.dump(percentages, f, indent=2, sort_keys=True)
        
def compute_and_save_discussed_stats_per_pitfall(df, pitfalls, output_dir):
    """
    For each pitfall (and overall), compute how often labels related to 'Present', 'Likely present', 'Partly present'
    are discussed vs not discussed. Saves absolute and percentage stats as JSON.
    """
    base_labels = ['Present', 'Likely present', 'Partly present']
    result = {}
    
    total_discussed = 0
    total_not_discussed = 0

    for pitfall_idx, pitfall_name in enumerate(pitfalls):
        pitfall_column = "Pitfall applies?" if pitfall_idx == 0 else f"Pitfall applies?.{pitfall_idx}"
        discussed = 0
        not_discussed = 0

        for i in range(0, len(df), 4):
            label = df.iloc[i + 3][pitfall_column]
            if label is None:
                continue
            for base_label in base_labels:
                if label.startswith(base_label):
                    if '(but discussed)' in label:
                        discussed += 1
                        total_discussed += 1
                    else:
                        not_discussed += 1
                        total_not_discussed += 1
                    break

        total = discussed + not_discussed
        percentage = round((discussed / total) * 100, 2) if total > 0 else 0.0

        result[pitfall_name] = {
            'discussed': discussed,
            'not_discussed': not_discussed,
            'total': total,
            'percentage_discussed': percentage
        }

    # Add total across all pitfalls
    total_count = total_discussed + total_not_discussed
    total_percentage = round((total_discussed / total_count) * 100, 2) if total_count > 0 else 0.0
    result['total'] = {
        'discussed': total_discussed,
        'not_discussed': total_not_discussed,
        'total': total_count,
        'percentage_discussed': total_percentage
    }

    with open(output_dir / 'discussed_stats_per_pitfall.json', 'w') as f:
        json.dump(result, f, indent=2, sort_keys=True)
        
def plot_pitfall_final_value_counts(pitfall_stats, order_of_labels, output_dir):
    """Plot stacked bar chart of final pitfall value percentages per pitfall."""
    df = pd.DataFrame(pitfall_stats).fillna(0).astype(int).T
    df_percent = df.div(df.sum(axis=1), axis=0) * 100
    df_percent = df_percent.sort_index(ascending=False).reindex(columns=reversed(order_of_labels)) # type: ignore
    
    fig, ax = plt.subplots(figsize=(24, 10))
    df_percent.plot(kind='barh', stacked=True, colormap='vlag', ax=ax)
    plt.title('Pitfall Final Value Percentages per Pitfall')
    plt.xlabel('Percentage (%)')
    plt.ylabel('Pitfall')
    legend = plt.legend(title='Final Value', bbox_to_anchor=(1.01, 1.01), loc='upper left')

    print("Legend colors:")
    for text, handle in zip(legend.get_texts(), legend.legend_handles):
        color = to_hex(handle.get_facecolor()) # type: ignore
        print(f"- {text.get_text()}: {color}")

    plt.tight_layout(rect=(0, 0, 0.85, 1))
    fig.savefig(output_dir / 'pitfalls.svg', format='svg', bbox_inches='tight')

def plot_reviewer_agreement(reviewer_agreement, output_dir):
    """Plot pie chart of reviewer agreement vs disagreement."""
    plt.figure(figsize=(5, 5))
    plt.pie(
        reviewer_agreement.values(),
        labels=reviewer_agreement.keys(),
        autopct='%1.1f%%',
        startangle=90,
        colors=['#66b3ff', '#ff9999']
    )
    plt.title('Reviewer Agreement vs Disagreement')
    plt.tight_layout()
    plt.savefig(output_dir / 'reviewer_agreement.svg', format='svg')

def plot_topic_distribution(topic_counts, output_dir):
    """Plot bar chart showing number of papers per topic."""
    topic_df = pd.DataFrame(topic_counts.items(), columns=['topic', 'count']).sort_values('count', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='topic', y='count', hue='topic', data=topic_df, palette='viridis', legend=False)
    plt.title('Number of Papers per Topic')
    plt.ylabel('Number of Papers')
    plt.xlabel('Topic')
    plt.tight_layout()
    plt.savefig(output_dir / 'topics.svg', format='svg', bbox_inches='tight')

def plot_pitfall_disagreement(pitfall_disagreement, output_dir):
    """Plot bar chart of reviewer disagreement per pitfall."""
    dis_df = pd.DataFrame.from_dict(pitfall_disagreement, orient='index', columns=['disagreements']).sort_values('disagreements', ascending=False)
    dis_df['pitfall'] = dis_df.index
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='pitfall', y='disagreements', hue='pitfall', data=dis_df, palette='rocket', legend=False)
    plt.title('Reviewer Disagreement per Pitfall')
    plt.ylabel('Number of Disagreements')
    plt.xlabel('Pitfall')
    plt.xticks(rotation=75, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'pitfall_disagreement.svg', format='svg', bbox_inches='tight')

def plot_pitfall_kappa(pitfall_kappa, output_dir):
    """Plot bar chart of Fleiss' Kappa values per pitfall."""
    if not pitfall_kappa:
        return
        
    kappa_df = pd.DataFrame.from_dict(pitfall_kappa, orient='index', columns=['kappa']).sort_values('kappa', ascending=False)
    kappa_df['pitfall'] = kappa_df.index
    
    plt.figure(figsize=(12, 6))
    
    # Color mapping for Kappa interpretation
    color_map = [(0, '#d62728'), (0.2, '#ff7f0e'), (0.4, '#ffb347'), 
                 (0.6, '#98df8a'), (0.8, '#2ca02c'), (1.0, '#1f7a1f')]
    colors = []
    for kappa in kappa_df['kappa']:
        for threshold, color in color_map:
            if kappa < threshold:
                colors.append(color)
                break
        else:
            colors.append('#1f7a1f')
    
    plt.bar(range(len(kappa_df)), kappa_df['kappa'], color=colors)
    plt.title('Inter-Reviewer Agreement (Fleiss\' Kappa) per Pitfall')
    plt.ylabel('Fleiss\' Kappa')
    plt.xlabel('Pitfall')
    plt.xticks(range(len(kappa_df)), kappa_df['pitfall'], rotation=75, ha='right') # type: ignore

    plt.ylim(-0.2, 1.0)
    plt.tight_layout()
    plt.savefig(output_dir / 'pitfall_kappa.svg', format='svg', bbox_inches='tight')

def save_stats_json(stats, output_dir):
    """Save statistics as JSON file."""
    with open(output_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2, sort_keys=True)

def main():
    script_dir = Path(__file__).parent.absolute()
    
    PITFALLS_CSV = script_dir / 'data' / 'pitfalls.csv'
    ORDER_OF_LABELS = [
        'Present', 'Present (but discussed)', 'Partly present', 'Partly present (but discussed)',
        'Likely present', 'Likely present (but discussed)', 'Does not apply', 'Unclear from text', 'Not present'
    ]
    HIDDEN_ROWS = [
        43, 52, 57, 58, 59, 68, 69, 70, 147, 148, 149, 150, 151, 152, 185, 210, 215, 216, 217, 230, 239, 240, 261, 262, 263, 304, 305, 306
    ]
    
    output_dir = script_dir / 'output'
    output_dir.mkdir(exist_ok=True)
    
    df, pitfalls = load_and_clean_data(PITFALLS_CSV, HIDDEN_ROWS)
    stats = aggregate_stats(df, pitfalls)
    
    plot_pitfall_final_value_counts(stats['pitfall_stats'], ORDER_OF_LABELS, output_dir)
    plot_reviewer_agreement(stats['reviewer_agreement'], output_dir)
    plot_topic_distribution(stats['topic_counts'], output_dir)
    plot_pitfall_disagreement(stats['pitfall_disagreement'], output_dir)
    plot_pitfall_kappa(stats['pitfall_kappa'], output_dir)
    save_stats_json(stats, output_dir)
    compute_and_save_pitfall_percentages_per_topic(df, pitfalls, output_dir)
    compute_and_save_present_percentages_per_topic(df, pitfalls, output_dir)
    compute_and_save_discussed_stats_per_pitfall(df, pitfalls, output_dir)
    
    print(f"All outputs saved to {output_dir}/")

if __name__ == "__main__":
    main()