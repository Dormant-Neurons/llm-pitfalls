import pandas as pd

from analysis.main import aggregate_stats, load_and_clean_data, calculate_fleiss_kappa_for_pitfall


def make_test_csv(tmp_path):
    # Test CSV with double header, 3 pitfalls, a hidden pitfall, and some summary/footer rows at the bottom
    csv_content = (
        'Definitions for each pitfall can be found here.,Reviewer,,,,,Paper is relevant?,,P1. Model Collapse via Synthetic Training Data,,P2. Data Leakage for LLMs,,P3. Too Small Context Size,,,,,Hidden Pitfall,,\n'
        'Paper Title,A,B,Date when the paper was accessed,Discussed when?,Topic,Check if LLMs are really an essential part of the paper,"If No, why?",Pitfall applies?,"If yes, why?",Pitfall applies?,"If yes, why?",Pitfall applies?,"If yes, why?",,,,,Hidden applies?,"If yes, why?"\n'
        # Paper 1 (rows 3-6)
        'Paper One,Ann,Bob,2024-01-01,No,TopicA,Yes,,Present,,Not present,,Present,,,,,HiddenValue,,\n'
        'Paper One,Ann,Bob,2024-01-01,No,TopicA,Yes,,Present,,Present,,Present,,,,,HiddenValue,,\n'
        'Paper One,Ann,Bob,2024-01-01,No,TopicA,Yes,,Partly present,,Not present,,Partly present,,,,,HiddenValue,,\n'
        'Paper One,Ann,Bob,2024-01-01,No,TopicA,Yes,,Partly present,,Not present,,Partly present,,,,,HiddenValue,,\n'
        # Hidden pitfall (rows 7-10)
        'Paper Two,Ann,Bob,2024-01-01,No,TopicA,No,,Present,,Not present,,Present,,,,,HiddenValue,,\n'
        'Paper Two,Ann,Bob,2024-01-01,No,TopicA,No,,Present,,Present,,Present,,,,,HiddenValue,,\n'
        'Paper Two,Ann,Bob,2024-01-01,No,TopicA,No,,Present,,Not present,,Partly present,,,,,HiddenValue,,\n'
        'Paper Two,Ann,Bob,2024-01-01,No,TopicA,No,,Partly present,,Not present,,Partly present,,,,,HiddenValue,,\n'
        # Paper 2 (rows 11-14)
        'Paper Three,Chris,Deb,2024-01-02,Yes,TopicB,No,,Not present,,Present,,Not present,,,,,HiddenValue,,\n'
        'Paper Three,Chris,Deb,2024-01-02,Yes,TopicB,No,,Not present,,Present,,Not present,,,,,HiddenValue,,\n'
        'Paper Three,Chris,Deb,2024-01-02,Yes,TopicB,No,,Not present,,Present,,Not present,,,,,HiddenValue,,\n'
        'Paper Three,Chris,Deb,2024-01-02,Yes,TopicB,No,,Not present,,Present,,Not present,,,,,HiddenValue,,\n'
        # Hidden pitfall (rows 15)
        'Paper Four,Chris,Deb,2024-01-02,No,TopicC,No,,Present,,Present,,Present,,,,,HiddenValue,,\n'
        # Paper 3 (rows 16-19)
        'Paper Five,Chris,Deb,2024-01-02,No,TopicC,No,,Present,,Not Present,,Partly Present,,,,,HiddenValue,,\n'
        'Paper Five,Chris,Deb,2024-01-02,No,TopicC,No,,Not Present,,Present,,Present,,,,,HiddenValue,,\n'
        'Paper Five,Chris,Deb,2024-01-02,No,TopicC,No,,Present,,Present,,Present,,,,,HiddenValue,,\n'
        'Paper Five,Chris,Deb,2024-01-02,No,TopicC,No,,Present,,Present,,Present,,,,,HiddenValue,,\n'
        # Summary/footer rows at the bottom
        ',,,,,,,,,,,,,,,,,\n'
        ',,,,,,,,,,,,,,,,,\n'
        '6.5.2025,,,,,,,,,,,,,,,,\n'
        '55 paper,,,,,,,,,,,,,,,,\n'
        '27 mit einem review,,,,,,,,,,,,,,,,\n'
        '14 mit zwei reviews,,,,,,,,,,,,,,,,\n'
        ',,,,,,,,,,,,,,,,,\n'
    )
    csv_path = tmp_path / 'pitfalls.csv'
    with open(csv_path, 'w') as f:
        f.write(csv_content)
    return str(csv_path)

def test_load_and_clean_and_aggregate(tmp_path):
    csv_path = make_test_csv(tmp_path)
    hidden_rows = [7, 8, 9, 10, 15]
    df, pitfalls = load_and_clean_data(csv_path, hidden_rows)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 12
    assert isinstance(pitfalls, list)
    # Only the first 3 pitfalls should be processed
    assert pitfalls == ['P1. Model Collapse via Synthetic Training Data', 'P2. Data Leakage for LLMs', 'P3. Too Small Context Size']
    stats = aggregate_stats(df, pitfalls)
    pitfall_stats = stats['pitfall_stats']
    reviewer_agreement = stats['reviewer_agreement']
    topic_counts = stats['topic_counts']
    pitfall_disagreement = stats['pitfall_disagreement']
    # Check that both papers and both topics are present
    assert set(df['Paper Title']) == {'Paper One', 'Paper Three', 'Paper Five'}
    assert set(df['Topic']) == {'TopicA', 'TopicB', 'TopicC'}
    # Check that pitfall_stats has only the 3 pitfalls
    assert set(pitfall_stats.keys()) == set(pitfalls)
    print(pitfall_stats)
    # Check actual stats for P1
    assert pitfall_stats['P1. Model Collapse via Synthetic Training Data']['Partly present'] == 1
    assert pitfall_stats['P1. Model Collapse via Synthetic Training Data']['Not present'] == 1
    assert pitfall_stats['P1. Model Collapse via Synthetic Training Data']['Present'] == 1
    # For P2
    assert pitfall_stats['P2. Data Leakage for LLMs']['Present'] == 2
    assert pitfall_stats['P2. Data Leakage for LLMs']['Not present'] == 1
    # For P3
    assert pitfall_stats['P3. Too Small Context Size']['Partly present'] == 1
    assert pitfall_stats['P3. Too Small Context Size']['Not present'] == 1
    assert pitfall_stats['P3. Too Small Context Size']['Present'] == 1
    # Reviewer agreement
    assert reviewer_agreement['agree'] == 5
    assert reviewer_agreement['disagree'] == 4
    # Topic counts - should have 1 paper for each topic
    assert topic_counts['TopicA'] == 1
    assert topic_counts['TopicB'] == 1
    assert topic_counts['TopicC'] == 1
    assert len(topic_counts) == 3
    # Pitfall disagreement should be a dict with keys for each pitfall
    assert set(pitfall_disagreement.keys()) == set(pitfalls)
    assert all(isinstance(v, int) for v in pitfall_disagreement.values())
    # Check actual disagreement values for each pitfall
    # These values are based on the test CSV data above
    assert pitfall_disagreement['P1. Model Collapse via Synthetic Training Data'] == 1
    assert pitfall_disagreement['P2. Data Leakage for LLMs'] == 2
    assert pitfall_disagreement['P3. Too Small Context Size'] == 1
    
    # Check that kappa values are calculated and returned
    pitfall_kappa = stats['pitfall_kappa']
    assert set(pitfall_kappa.keys()) == set(pitfalls)
    assert all(isinstance(v, (int, float)) for v in pitfall_kappa.values())
    # Kappa values should be between -1 and 1
    assert all(-1 <= v <= 1 for v in pitfall_kappa.values())
    
    # Check that overall kappa is calculated and returned
    overall_kappa = stats['overall_kappa']
    assert isinstance(overall_kappa, (int, float))
    # Overall kappa should be between -1 and 1
    assert -1 <= overall_kappa <= 1


def test_fleiss_kappa_calculation():
    """Test the Fleiss' Kappa calculation function with known test cases."""
    
    # Test case 1: Perfect agreement
    perfect_agreement = [
        ['Present', 'Present'],
        ['Not present', 'Not present']
    ]
    kappa_perfect = calculate_fleiss_kappa_for_pitfall(perfect_agreement)
    assert kappa_perfect == 1.0, f"Expected perfect agreement (1.0), got {kappa_perfect}"
    
    # Test case 2: Complete disagreement
    no_agreement = [
        ['Present', 'Not present'],
        ['Not present', 'Present']
    ]
    kappa_none = calculate_fleiss_kappa_for_pitfall(no_agreement)
    assert kappa_none == -1.0, f"Expected complete disagreement (-1.0), got {kappa_none}"
    
    # Test case 3: Known exact kappa value
    # 6 subjects: 4 agreements, 2 disagreements
    # P_observed = 4/6 = 2/3, P_expected = 0.5, kappa = (2/3 - 0.5) / (1 - 0.5) = 1/3
    known_case = [
        ['A', 'A'],  # Agreement
        ['A', 'B'],  # Disagreement  
        ['B', 'B'],  # Agreement
        ['A', 'A'],  # Agreement
        ['B', 'A'],  # Disagreement
        ['B', 'B']   # Agreement
    ]
    kappa_known = calculate_fleiss_kappa_for_pitfall(known_case)
    expected_kappa = 1/3  # Exactly 0.333...
    assert abs(kappa_known - expected_kappa) < 0.01, f"Expected kappa ~= {expected_kappa:.3f}, got {kappa_known}"
