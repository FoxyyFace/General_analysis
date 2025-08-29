"""
if YOU need to calculate someone identify outliers,please write a function to calculate the outliers
i don't write this function, because in addtion to the paper:

G.R. Lewin, A. Kapur, D.M. Cornforth, R.P. Duncan, F.L. Diggle, D.A. Moustafa, S.A. Harrison, E.P. Skaar, W.J. Chazin, J.B. Goldberg, J.M. Bomberger, & M. Whiteley, 
Application of a quantitative framework to improve the accuracy of a bacterial infection model, 
Proc. Natl. Acad. Sci. U.S.A. 120 (19) e2221542120, 
https://doi.org/10.1073/pnas.2221542120 (2023).

The ROUT method in GraphPad Prism was used to identify outliers, which identified sample Rn_PAO1_SCFM2_15 as an outlier; 
this sample was excluded from all further analyses.
"""

"""
Output Directory Structure Documentation:

1. mean_sd/
   - Stores mean expression values and standard deviations
   - Files:
     * {group_name}_mean_expression.csv: Mean gene expression per experimental group
     * nature_sd.csv: Standard deviations of gene expression in natural samples

2. Z-scores/
   - Contains Z-score calculations comparing experimental groups to natural samples
   - Files:
     * {group_name}_zscore.csv: Z-scores for each gene (formula: (group_mean - nature_mean)/nature_sd)

3. Mean_Z_score_AS2/
   - Stores Accuracy Scores (AS2) calculated from group means
   - Files:
     * as2_scores.csv: Percentage of genes with |Z-score| < 2 per group (direct group-level calculation)

4. sample_as2_scores/
   - Contains sample-level AS2 calculations and group statistics
   - Files:
     * sample_as2_scores.csv: AS2 scores for individual samples
     * group_as2_statistics.csv: Aggregated statistics (mean, std, count) per group

5. borderline_gene/
   - Stores genes near the AS2 threshold (|Z-score| ≈ 2)
   - Files:
     * {group_name}_upper_border.csv: Genes slightly above threshold (1.5 < Z < 2.5)
     * {group_name}_lower_border.csv: Genes slightly below threshold (-2.5 < Z < -1.5)

Analysis Methodology:
- Two approaches for AS2 calculation (as per PNAS paper):
  1. Direct group-level: Compare group means to nature (Mean_Z_score_AS2/)
  2. Sample-level: Calculate per-sample AS2 then average by group (sample_as2_scores/)
- Borderline gene analysis identifies genes most sensitive to model conditions
- All outputs are CSV format for compatibility with downstream analysis tools

Visualization:
- plot_borderline_genes(): Generates distribution plots of borderline genes
- plot_group_as2_distribution(): Creates boxplots of sample-level AS2 by group
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def process_counts_by_condition_and_group(metadata_df, counts_df, data_dir):
    """
    Process counts data based on conditions and groups from metadata.
    
    Parameters:
    -----------
    metadata_df : pandas.DataFrame
        DataFrame containing metadata (should have 'sample_names', 'condition', 'group' columns)
    counts_df : pandas.DataFrame
        Counts DataFrame (genes as rows, samples as columns)
    
    Returns:
    --------
    dict
        A dictionary containing:
        - 'nature': DataFrame with only nature samples
        - 'samples_by_group': Dictionary of DataFrames keyed by group names
    """
    # Convert sample names to sets for comparison
    counts_samples = set(counts_df.columns)
    meta_samples = set(metadata_df['sample_names'])
    
    # Find samples that are in counts but not in metadata
    missing_in_meta = counts_samples - meta_samples
    if missing_in_meta:
        logger.warning(f"{len(missing_in_meta)} samples in counts but not in metadata: {missing_in_meta}")
    
    # Find samples that are in metadata but not in counts
    missing_in_counts = meta_samples - counts_samples
    if missing_in_counts:
        logger.warning(f"{len(missing_in_counts)} samples in metadata but not in counts: {missing_in_counts}")
    
    # Initialize result dictionary
    result = {
        'nature': None,
        'samples_by_group': {}
    }
    
    # 1. Separate nature samples
    nature_samples = metadata_df[metadata_df['condition'] == 'nature']['sample_names']
    # Convert to list and filter only samples present in counts data
    valid_nature_samples = [s for s in nature_samples if s in counts_df.columns]
    if valid_nature_samples:
        result['nature'] = counts_df[valid_nature_samples]
        logger.info(f"Found {len(valid_nature_samples)} nature samples")
    else:
        logger.warning("No nature samples found in counts data")
    
    # 2. Group samples by their group (only for samples condition)
    samples_metadata = metadata_df[metadata_df['condition'] == 'samples']
    
    for group_name, group_df in samples_metadata.groupby('group'):
        group_samples = group_df['sample_names']
        # Filter samples that exist in counts data
        valid_samples = [s for s in group_samples if s in counts_df.columns]
        
        if valid_samples:
            result['samples_by_group'][group_name] = counts_df[valid_samples]
            logger.info(f"Group {group_name}: {len(valid_samples)} valid samples")
        else:
            logger.warning(f"No valid samples found for group {group_name} in counts data")

    # Calculate group means
    all_groups = {'nature': result['nature'], **result['samples_by_group']}
    group_means = calculate_group_means(all_groups)
    
    # Combine group means into the result
    result['group_means'] = group_means
    
    logger.info("Group means calculated")
    
    # Calculate standard deviation for nature group
    if result['nature'] is not None:
        nature_sd = calculate_group_sd(result['nature'])
        result['nature_sd'] = nature_sd
        logger.info("Nature group standard deviations calculated")
    else:
        logger.warning("Cannot calculate SD for nature group - no data available")
    
    # Add self-AS2 calculation for nature samples
    if result['nature'] is not None:
        nature_samples = result['nature'].columns.tolist()
        self_as2_scores = {}
        
        # Strictly follow the 1,200 iterations as specified in the paper
        logger.info("Starting self-AS2 calculations for nature group (1,200 iterations)...")
        for i in range(1200): 
            if i % 100 == 0: 
                logger.info(f"Progress: {i/1200 * 100:.1f}%")
            model_samples = np.random.choice(nature_samples, 3, replace=False)
            target_samples = [s for s in nature_samples if s not in model_samples]
            
            temp_z = calculate_z_scores(
                {'model': result['nature'][model_samples].mean(axis=1).to_frame('model_mean')},
                result['nature'][target_samples].mean(axis=1).to_frame('nature_mean'),
                result['nature'][target_samples].std(axis=1).to_frame('nature_sd')
            )
            self_as2_scores[f'iter_{i}'] = calculate_as2(temp_z)['model']
        
        result['self_as2'] = pd.DataFrame.from_dict(
            self_as2_scores, orient='index', columns=['AS2']
        )

        nature_self_dir = os.path.join(data_dir, "nature_self_as2")
        os.makedirs(nature_self_dir, exist_ok=True)
        result['self_as2'].to_csv(os.path.join(nature_self_dir, "self_as2_iterations.csv"))

        # Save statistics
        stats_content = f"""Nature Self-AS2 Statistical Summary (Paper Method)
===============================================
Total Iterations: {len(self_as2_scores)}
Mean AS2: {result['self_as2'].mean().values[0]:.2f}%
Standard Deviation: {result['self_as2'].std().values[0]:.2f}
Minimum AS2: {result['self_as2'].min().values[0]:.2f}%
Maximum AS2: {result['self_as2'].max().values[0]:.2f}%
95% Confidence Interval: [{result['self_as2'].mean().values[0] - 1.96*result['self_as2'].std().values[0]:.2f}%, 
                     {result['self_as2'].mean().values[0] + 1.96*result['self_as2'].std().values[0]:.2f}%]
"""
        stats_file = os.path.join(nature_self_dir, "self_as2_stats.txt")
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write(stats_content)
        
        logger.info(f"Self-AS2 results saved to {nature_self_dir}")

    z_scores = calculate_z_scores(
        result['group_means'],
        result['group_means']['nature'],
        result['nature_sd']
    )
    result['z_scores'] = z_scores
    
    logger.info("Z-score calculation completed")
    
    return result

def calculate_group_means(counts_dict):
    """
    Calculate mean expression for each gene across samples in each group
    
    Parameters:
    -----------
    counts_dict : dict
        Dictionary containing DataFrames for each group
    
    Returns:
    --------
    dict
        Dictionary with group names as keys and mean expression DataFrames as values
    """
    group_means = {}
    for group_name, group_df in counts_dict.items():
        if isinstance(group_df, pd.DataFrame) and not group_df.empty:
            # Calculate mean across columns (samples) for each gene
            group_means[group_name] = group_df.mean(axis=1).to_frame(name=f"{group_name}_mean")
    return group_means

def calculate_group_sd(counts_data):
    """
    Calculate standard deviation for each gene in nature group
    
    Parameters:
    -----------
    counts_data : pandas.DataFrame
        DataFrame containing expression data for nature group
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with gene standard deviations
    """
    if counts_data is None or counts_data.empty:
        logger.warning("No nature samples data available for SD calculation")
        return None
    
    # Calculate standard deviation across samples for each gene
    nature_sd = counts_data.std(axis=1, ddof=1).to_frame(name="nature_sd")
    return nature_sd

def calculate_z_scores(group_means_dict, nature_mean, nature_sd):
    """
    Calculate Z-scores for all groups compared to nature group
    
    Parameters:
    -----------
    group_means_dict : dict
        Dictionary containing mean expression DataFrames for each group
    nature_mean : pandas.DataFrame
        Mean expression of nature group
    nature_sd : pandas.DataFrame
        Standard deviation of nature group
    
    Returns:
    --------
    dict
        Dictionary with group names as keys and Z-score DataFrames as values
    """
    z_scores = {}
    
    # Check if nature_mean and nature_sd are provided
    if nature_mean is None or nature_sd is None:
        logger.warning("Cannot calculate Z-scores - missing nature group data")
        return z_scores
    
    for group_name, group_mean in group_means_dict.items():
        # Skip nature group as we don't compare it to itself
        if group_name == 'nature':
            continue
            
        # Z-score: (group_mean - nature_mean) / nature_sd
        z_score = (group_mean.iloc[:, 0] - nature_mean.iloc[:, 0]) / nature_sd.iloc[:, 0]
        z_scores[group_name] = z_score.to_frame(f"{group_name}_zscore")
    
    return z_scores

def calculate_as2(z_scores_dict):
    """
    Calculate AS2 score for each group (percentage of genes with |Z-score| < 2)
    
    Parameters:
    -----------
    z_scores_dict : dict
        Dictionary containing Z-score DataFrames for each group
    
    Returns:
    --------
    dict
        Dictionary with group names as keys and AS2 scores as values
    """
    as2_scores = {}
    
    for group_name, z_df in z_scores_dict.items():
        total_genes = len(z_df)
        if total_genes == 0:
            as2_scores[group_name] = 0.0
            continue
            
        # Count genes with |Z-score| < 2
        stable_genes = sum(abs(z_df.iloc[:, 0]) < 2)
        as2_score = (stable_genes / total_genes) * 100
        as2_scores[group_name] = as2_score
    
    return as2_scores

def calculate_sample_as2(counts_df, metadata_df, nature_mean, nature_sd):
    """
    Calculate AS2 score for each individual sample compared to nature group
    
    Parameters:
    -----------
    counts_df : pandas.DataFrame
        Counts DataFrame (genes as rows, samples as columns)
    metadata_df : pandas.DataFrame
        Metadata DataFrame containing sample information
    nature_mean : pandas.DataFrame
        Mean expression of nature group
    nature_sd : pandas.DataFrame
        Standard deviation of nature group
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with sample names as index and AS2 scores as values
    """
    if nature_mean is None or nature_sd is None:
        logger.warning("Cannot calculate sample AS2 - missing nature group data")
        return None
    
    sample_as2 = {}
    
    # Get list of non-nature samples
    sample_list = metadata_df[metadata_df['condition'] != 'nature']['sample_names']
    
    for sample in sample_list:
        if sample not in counts_df.columns:
            continue
            
        # Calculate Z-score for this sample: (sample - nature_mean) / nature_sd
        z_scores = (counts_df[sample] - nature_mean.iloc[:, 0]) / nature_sd.iloc[:, 0]
        
        # Calculate AS2: percentage of genes with |Z| < 2
        total_genes = len(z_scores)
        stable_genes = sum(abs(z_scores) < 2)
        as2_score = (stable_genes / total_genes) * 100 if total_genes > 0 else 0
        
        sample_as2[sample] = as2_score
    
    return pd.DataFrame.from_dict(sample_as2, orient='index', columns=['AS2_score'])

def analyze_borderline_genes(z_scores, threshold=2, margin=0.5):
    """
    Analyze borderline genes (key diagnostic points as described in the paper)
    
    Parameters:
    z_scores (dict): Dictionary of Z-score DataFrames for each group
    threshold (float): AS2 threshold (default 2)
    margin (float): Borderline margin (default ±0.5)
    
    Returns:
    dict: Results of borderline gene analysis
    """
    results = {}
    
    for group_name, z_df in z_scores.items():
        # Extract the z-score for the current group
        z_vals = z_df.iloc[:, 0]
        
        # Identify borderline genes
        borderline_mask = (abs(z_vals) > (threshold - margin)) & (abs(z_vals) < (threshold + margin))
        borderline_genes = z_df[borderline_mask]
        
        # Classify borderline genes (upper/lower border)
        upper_border = borderline_genes[borderline_genes.iloc[:,0] > 0]
        lower_border = borderline_genes[borderline_genes.iloc[:,0] < 0]
        
        results[group_name] = {
            'total_genes': len(z_vals),
            'borderline_count': len(borderline_genes),
            'borderline_percent': len(borderline_genes)/len(z_vals)*100,
            'upper_border': upper_border,
            'lower_border': lower_border
        }
        
        logger.info(f"{group_name}: {len(borderline_genes)} borderline genes ({len(borderline_genes)/len(z_vals)*100:.2f}%)")
    
    return results

def plot_borderline_genes(borderline_results, output_dir):
    """Visualize the distribution of borderline genes"""
    for group, data in borderline_results.items():
        plt.figure(figsize=(10, 6))
        # Upper border genes
        if not data['upper_border'].empty:
            sns.kdeplot(data['upper_border'].iloc[:, 0], label='Upper Border', fill=True)
        # Lower border genes
        if not data['lower_border'].empty:
            sns.kdeplot(data['lower_border'].iloc[:, 0], label='Lower Border', fill=True)
            
        plt.axvline(x=2, color='r', linestyle='--', alpha=0.5)
        plt.axvline(x=-2, color='r', linestyle='--', alpha=0.5)
        plt.title(f"Borderline Genes Distribution: {group}")
        plt.xlabel("Z-score")
        plt.legend()
        
        # Save plot
        output_file = os.path.join(output_dir, f"borderline_distribution_{group}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved borderline gene distribution plot to {output_file}")

def calculate_group_as2_from_samples(sample_as2_df, metadata_df):
    """
    Calculate group average AS2 from sample-level AS2 scores
    
    Parameters:
    sample_as2_df : pd.DataFrame
        DataFrame containing AS2 scores for all samples
    metadata_df : pd.DataFrame
        Metadata DataFrame containing sample group information
    
    Returns:
    dict
        Mapping from group name to average AS2 score
    """
    # Merge AS2 scores with group information
    grouped_as2 = sample_as2_df.join(
        metadata_df.set_index('sample_names')['group'],
        how='left'
    )
    
    # Calculate average AS2 for each group
    group_avg_as2 = grouped_as2.groupby('group')['AS2_score'].mean()
    
    return group_avg_as2.to_dict()

def plot_group_as2_distribution(sample_as2_df, metadata_df, output_dir):
    """Plot boxplots of AS2 score distribution for each group"""
    # Merge data
    plot_data = sample_as2_df.join(
        metadata_df.set_index('sample_names')['group'],
        how='inner'
    )
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=plot_data, x='group', y='AS2_score')
    plt.title("AS2 Score Distribution by Group")
    plt.ylabel("AS2 Score (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, "AS2_distribution.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved AS2 distribution plot to {output_file}")

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='AS2 Analysis Tool for Bacterial Infection Models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/output parameters
    parser.add_argument('--input_dir', default='output', 
                        help='Input directory containing VST data and metadata')
    parser.add_argument('--output_dir', default='AS2_results', 
                        help='Output directory for all analysis results')
    
    # File names
    parser.add_argument('--vst_file', default='results_counts_vst.csv', 
                        help='File name of VST-transformed data')
    parser.add_argument('--metadata_file', default='metadata_df.csv', 
                        help='File name of sample metadata')
    
    # Analysis parameters
    parser.add_argument('--borderline_margin', type=float, default=0.5, 
                        help='Margin for borderline gene identification')
    
    # Logging
    parser.add_argument('--log_level', default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging verbosity level')
    
    return parser.parse_args()

def main(args):
    # Set up directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create subdirectories for analysis results
    mean_sd_dir = os.path.join(args.output_dir, "mean_sd")
    z_score_dir = os.path.join(args.output_dir, "Z-scores")
    mean_as2_dir = os.path.join(args.output_dir, "Mean_Z_score_AS2")
    sample_as2_dir = os.path.join(args.output_dir, "sample_as2_scores")
    borderline_dir = os.path.join(args.output_dir, "borderline_gene")
    
    for subdir in [mean_sd_dir, z_score_dir, mean_as2_dir, sample_as2_dir, borderline_dir]:
        os.makedirs(subdir, exist_ok=True)
    
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"VST file: {os.path.join(args.input_dir, args.vst_file)}")
    logger.info(f"Metadata file: {os.path.join(args.input_dir, args.metadata_file)}")
    
    # Load data
    try:
        counts_df_by_vst = pd.read_csv(os.path.join(args.input_dir, args.vst_file), index_col=0)
        metadata_df = pd.read_csv(os.path.join(args.input_dir, args.metadata_file))
        
        logger.info(f"Loaded VST data: {counts_df_by_vst.shape[0]} genes, {counts_df_by_vst.shape[1]} samples")
        logger.info(f"Loaded metadata: {len(metadata_df)} samples")
    except Exception as e:
        logger.error(f"Failed to load input files: {str(e)}")
        return
    
    # Run AS2 analysis
    try:
        result = process_counts_by_condition_and_group(metadata_df, counts_df_by_vst, args.output_dir)
        
        # Save results
        # 1. Group means
        if 'group_means' in result:
            for group_name, mean_df in result['group_means'].items():
                output_file = os.path.join(mean_sd_dir, f"{group_name}_mean_expression.csv")
                mean_df.to_csv(output_file)
                logger.info(f"Saved group means for {group_name} to {output_file}")
        else:
            logger.warning("No group means to save")
        
        # 2. Nature standard deviation
        if 'nature_sd' in result and result['nature_sd'] is not None:
            output_file = os.path.join(mean_sd_dir, "nature_sd.csv")
            result['nature_sd'].to_csv(output_file)
            logger.info(f"Saved nature standard deviations to {output_file}")
        else:
            logger.warning("No nature SD to save")
        
        # 3. Z-scores
        if 'z_scores' in result:
            for group_name, z_df in result['z_scores'].items():
                output_file = os.path.join(z_score_dir, f"{group_name}_zscore.csv")
                z_df.to_csv(output_file)
                logger.info(f"Saved Z-scores for {group_name} to {output_file}")
        else:
            logger.warning("No Z-scores to save")
        
        # 4. AS2 scores from group means
        if 'z_scores' in result:
            as2_scores = calculate_as2(result['z_scores'])
            
            if as2_scores:
                as2_df = pd.DataFrame.from_dict(as2_scores, orient='index', columns=['AS2_score'])
                as2_file = os.path.join(mean_as2_dir, "as2_scores.csv")
                as2_df.to_csv(as2_file)
                logger.info(f"Saved group-level AS2 scores to {as2_file}")
                
                # Log AS2 scores
                logger.info("\nGroup-level AS2 Scores:")
                for group_name, score in as2_scores.items():
                    logger.info(f"{group_name}: {score:.2f}%")
        
        # 5. Sample-level AS2 calculations
        if ('group_means' in result and 'nature' in result['group_means'] and 
            result['group_means']['nature'] is not None and 
            'nature_sd' in result and result['nature_sd'] is not None):
            
            logger.info("Calculating sample-level AS2 scores...")
            sample_as2_df = calculate_sample_as2(
                counts_df_by_vst,
                metadata_df,
                result['group_means']['nature'],
                result['nature_sd']
            )
            
            if sample_as2_df is not None:
                # Save sample-level AS2 scores
                sample_as2_file = os.path.join(sample_as2_dir, "sample_as2_scores.csv")
                sample_as2_df.to_csv(sample_as2_file)
                logger.info(f"Saved sample-level AS2 scores to {sample_as2_file}")
                
                # Calculate group statistics
                if 'group' in metadata_df.columns:
                    grouped_as2 = sample_as2_df.join(metadata_df.set_index('sample_names')['group'])
                    group_stats = grouped_as2.groupby('group').agg(['mean', 'std', 'count'])
                    
                    if not group_stats.empty:
                        group_stats_file = os.path.join(sample_as2_dir, "group_as2_statistics.csv")
                        group_stats.to_csv(group_stats_file)
                        logger.info(f"Saved group-wise AS2 statistics to {group_stats_file}")
                        
                        # Calculate group averages using sample-level data
                        group_avg_as2 = calculate_group_as2_from_samples(sample_as2_df, metadata_df)
                        logger.info("\nGroup Average AS2 Scores (from sample-level data):")
                        for group, score in group_avg_as2.items():
                            logger.info(f"{group}: {score:.2f}%")
                
                # Create AS2 distribution plot
                plot_group_as2_distribution(sample_as2_df, metadata_df, args.output_dir)
        
        # 6. Borderline gene analysis
        if 'z_scores' in result:
            logger.info("Analyzing borderline genes...")
            borderline_results = analyze_borderline_genes(
                result['z_scores'],
                threshold=2,
                margin=args.borderline_margin
            )
            
            # Save borderline genes
            for group, data in borderline_results.items():
                upper_file = os.path.join(borderline_dir, f"{group}_upper_border.csv")
                lower_file = os.path.join(borderline_dir, f"{group}_lower_border.csv")
                data['upper_border'].to_csv(upper_file)
                data['lower_border'].to_csv(lower_file)
                logger.info(f"Saved borderline genes for {group} to {upper_file} and {lower_file}")
            
            # Visualize borderline genes
            plot_borderline_genes(borderline_results, args.output_dir)
        
        logger.info("AS2 analysis completed successfully!")
    except Exception as e:
        logger.error(f"AS2 analysis failed: {str(e)}")

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set logging level
    logger.setLevel(args.log_level)
    
    # Run main analysis
    main(args)