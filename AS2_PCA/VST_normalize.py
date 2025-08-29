import os
import argparse
from collections import defaultdict
import re
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def VST(counts_df, metadata_df, threads, aligned_samples=True):
    """
    Perform Variance Stabilizing Transformation (VST) on count data using DESeq2 methodology.
    
    Args:
        counts_df (pd.DataFrame): Raw count matrix (genes x samples)
        metadata_df (pd.DataFrame): Metadata with sample information
        threads (int): Number of CPU threads to use
        aligned_samples (bool): Whether to align samples between counts and metadata
        
    Returns:
        pd.DataFrame: VST-transformed count matrix
        pd.DataFrame: Normalized count matrix
    """
    # Print diagnostic information
    print("\n Columns in metadata_df:", metadata_df.columns.tolist())
    print(metadata_df.head().to_string(index=False))

    # Handle NaN values in count matrix
    if counts_df.isnull().any().any():
        print(f"警告：计数矩阵中发现 {counts_df.isnull().sum().sum()} 个 NaN 值")
        counts_df = counts_df.dropna(axis=0)  # Remove genes with NaN

    counts_df = counts_df.T  # Transpose to samples x genes

    print("\n Counts matrix shape (samples, genes):", counts_df.shape)
    print("Metadata shape:", metadata_df.shape)
    print("Metadata index:", metadata_df.index[:5])
    
    # Align samples if requested
    if aligned_samples:
        print("\n=== Running sample alignment ===")
        common_samples = counts_df.index.intersection(metadata_df.index)
        
        if len(common_samples) == 0:
            print("Warning: No common samples found between counts_df and metadata_df")
            print("Counts samples:", counts_df.index[:5].tolist())
            print("Metadata samples:", metadata_df.index[:5].tolist())
            print("=== Skipping sample alignment ===\n")
        else:
            counts_df = counts_df.loc[common_samples]
            metadata_df = metadata_df.loc[common_samples]
            print(f"Aligned {len(common_samples)} samples based on metadata index")
            print("Aligned count_samples index:", metadata_df.index[:5])
            print("Aligned metadata_samples index:", counts_df.index[:5])
    else:
        print("\n=== Skipping sample alignment ===")
    
    # Ensure consistent indexing
    if counts_df.index.tolist() != metadata_df.index.tolist():
        print("\nWarning: counts_df and metadata_df have different sample names.\nReindexing metadata_df to match counts_df index.")
        metadata_df = metadata_df.reindex(counts_df.index)
    
    # Check sample counts match
    if len(counts_df) != len(metadata_df):
        print(f"Warning: Number of samples differs (Counts: {len(counts_df)}, Metadata: {len(metadata_df)})")

    # Set up DESeq2 inference
    inference = DefaultInference(n_cpus=threads)    

    # Create DESeqDataSet
    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata_df,
        design="~condition",  # Model design formula
        refit_cooks=True,
        inference=inference,
    )

    # Fit size factors
    dds.fit_size_factors()
    
    # Fit genewise dispersions
    dds.fit_genewise_dispersions()
    
    # Fit dispersion trend
    dds.fit_dispersion_trend()
    
    # Fit dispersion prior
    dds.fit_dispersion_prior()
    print(f"logre_prior: {dds.uns['_squared_logres']}, sigma_prior: {dds.uns['prior_disp_var']}")

    # Perform VST
    dds.vst()

    # Extract VST-transformed counts
    counts_df_byvst = pd.DataFrame(
        dds.layers["vst_counts"],
        index=dds.obs_names,
        columns=dds.var_names
    ).T  # Transpose back to genes x samples

    # Extract normalized counts
    counts_df_bynormed = pd.DataFrame(
        dds.layers["normed_counts"],
        index=dds.obs_names,
        columns=dds.var_names
    ).T  # Transpose back to genes x samples

    return counts_df_byvst, counts_df_bynormed

def create_counts_df(natural_dir, samples_dir):
    """
    Build count matrix from featureCounts output files.
    
    Args:
        natural_dir (str): Path to natural environment samples
        samples_dir (str): Path to treated/variable samples
        
    Returns:
        pd.DataFrame: Count matrix (genes x samples)
    """
    all_data = defaultdict(list)
    sample_names = []
    print("Creating counts and metadata dataframes...")

    for root_dir in [natural_dir, samples_dir]:
        for subdir, dirs, files in os.walk(root_dir):
            if 'featureCounts' in dirs:
                featureCounts_dir = os.path.join(subdir, 'featureCounts')
                txt_files = [f for f in os.listdir(featureCounts_dir) if f.endswith('.txt')]

                if len(txt_files) == 1:
                    txt_path = os.path.join(featureCounts_dir, txt_files[0])
                    sample_dir = os.path.dirname(featureCounts_dir)
                    sample_name = os.path.basename(sample_dir)
                    sample_names.append(sample_name)
                    print(f"Reading file: {txt_path} -> Sample name: {sample_name}")

                    with open(txt_path, 'r') as f:
                        lines = f.readlines()

                    header_index = None
                    for i, line in enumerate(lines):
                        if line.startswith("Geneid"):
                            header_index = i
                            break

                    if header_index is not None:
                        for line in lines[header_index + 1:]:
                            parts = line.strip().split("\t")
                            if len(parts) < 7:
                                continue
                            gene_id = parts[0]
                            count = int(parts[-1])
                            all_data[gene_id].append(count)

    # Create DataFrame
    counts_df = pd.DataFrame.from_dict(all_data, orient='index', columns=sample_names)
    
    if len(sample_names) == len(counts_df.columns):
        print("Sample names match the number of columns in counts_df.")
    else:
        print("Warning: Sample names do not match the number of columns in counts_df.")

    print(f"\nSuccessfully read {len(sample_names)} samples for {len(counts_df)} genes.")
    print("Samples found:")
    print("\n".join(sample_names))

    return counts_df

def create_metadata(natural_dir, samples_dir, counts_df):
    """
    Create metadata DataFrame from sample directories.
    
    Args:
        natural_dir (str): Path to natural environment samples
        samples_dir (str): Path to treated/variable samples
        counts_df (pd.DataFrame): Count matrix (genes x samples)
        
    Returns:
        pd.DataFrame: Metadata with sample information
    """
    metadata = []
    all_samples = counts_df.columns.tolist()
    print("Sample names from counts:", all_samples[:5])
    
    # Create sample to info mapping
    sample_to_info = {}
    
    # Scan natural_dir
    for root, dirs, files in os.walk(natural_dir):
        if 'featureCounts' in dirs:
            dir_name = os.path.basename(root)
            base_name = re.sub(r'_\d+$', '', dir_name)
            sample_to_info[dir_name] = {'condition': 'nature'}
            sample_to_info[base_name] = {'condition': 'nature'}
    
    # Scan samples_dir
    for root, dirs, files in os.walk(samples_dir):
        if 'featureCounts' in dirs:
            dir_name = os.path.basename(root)
            base_name = re.sub(r'_\d+$', '', dir_name)
            sample_to_info[dir_name] = {'condition': 'samples'}
            sample_to_info[base_name] = {'condition': 'samples'}
    
    # Create metadata entries
    for sample in all_samples:
        base_name = re.sub(r'_\d+$', '', sample)
        
        if sample in sample_to_info:
            condition = sample_to_info[sample]['condition']
        elif base_name in sample_to_info:
            condition = sample_to_info[base_name]['condition']
        else:
            condition = 'nature'
            print(f"Warning: Can't find sample '{sample}' or '{base_name}' in directories")
        
        group = base_name
        metadata.append({
            'sample_names': sample,
            'condition': condition,
            'group': group
        })
    
    metadata_df = pd.DataFrame(metadata)
    print("Metadata samples:", metadata_df[['sample_names', 'condition']].head().to_string(index=False))
    
    return metadata_df

def filter_genes_by_expression(counts_df, metadata, min_samples_percent=0.95):
    """
    Filter genes based on expression in natural samples.
    
    Args:
        counts_df (pd.DataFrame): Count matrix (genes x samples)
        metadata (pd.DataFrame): Sample metadata
        min_samples_percent (float): Minimum percent of samples where gene must be expressed
        
    Returns:
        pd.DataFrame: Filtered count matrix
    """
    # Get natural samples
    sputum_samples = metadata[metadata['condition'] == 'nature']['sample_names']
    
    if len(sputum_samples) == 0:
        raise ValueError("No natural samples found! Check metadata condition column")
    
    # Calculate coverage in natural samples
    sputum_coverage = (counts_df[sputum_samples] > 0).mean(axis=1)
    
    # Apply filter
    expressed_genes = sputum_coverage[sputum_coverage >= min_samples_percent].index
    filtered_counts = counts_df.loc[expressed_genes]
    
    print(f"\nGene filtering based on natural samples:")
    print(f"Natural samples: {len(sputum_samples)}")
    print(f"Retained genes: {len(filtered_counts)}/{len(counts_df)}")
    print(f"Total samples after filtering: {len(filtered_counts.columns)}")
    
    return filtered_counts

def filter_samples_by_zeros(counts_df, max_zero_ratio=0.25):
    """
    Filter samples based on zero expression ratio.
    
    Args:
        counts_df (pd.DataFrame): Count matrix (genes x samples)
        max_zero_ratio (float): Maximum allowed zero ratio
        
    Returns:
        pd.DataFrame: Filtered count matrix
        pd.DataFrame: Info about filtered samples
    """
    # Calculate zero ratio per sample
    sample_zero_ratio = (counts_df == 0).mean(axis=0)
    
    # Identify low-quality samples
    low_quality_samples = sample_zero_ratio[sample_zero_ratio > max_zero_ratio].index.tolist()
    filtered_counts = counts_df.drop(columns=low_quality_samples)
    
    # Create filtered samples info
    filtered_info = pd.DataFrame({
        'sample': low_quality_samples,
        'zero_ratio': sample_zero_ratio[low_quality_samples],
        'reason': f'Zero ratio > {max_zero_ratio}'
    })
    
    # Print report
    total_genes = counts_df.shape[0]
    print(f"\nSample filtering (zero ratio threshold: {max_zero_ratio}):")
    print(f"Total samples: {counts_df.shape[1]}")
    print(f"Filtered samples: {len(low_quality_samples)}")
    print(f"Retained samples: {filtered_counts.shape[1]}")
    
    if low_quality_samples:
        print("\nFiltered sample details:")
        for sample in low_quality_samples:
            zero_count = (counts_df[sample] == 0).sum()
            print(f"{sample}: {zero_count}/{total_genes} zero genes ({zero_count/total_genes:.1%})")
    
    # Print filtering details
    print("\nSample filtering details:")
    print("="*50)
    for sample in counts_df.columns:
        zero_ratio = (counts_df[sample] == 0).mean()
        status = "Filtered" if zero_ratio > max_zero_ratio else "Retained"
        print(f"{sample}: Zero ratio={zero_ratio:.2%} → {status}")
    
    return filtered_counts, filtered_info

def validate_metadata(metadata_df, counts_df):
    """
    Ensure metadata matches count matrix samples.
    
    Args:
        metadata_df (pd.DataFrame): Sample metadata
        counts_df (pd.DataFrame): Count matrix (genes x samples)
        
    Returns:
        pd.DataFrame: Aligned metadata
        pd.DataFrame: Aligned count matrix
    """
    if 'sample_names' not in metadata_df.columns:
        metadata_df = metadata_df.reset_index().rename(columns={'index': 'sample_names'})
    
    metadata_samples = set(metadata_df['sample_names'])
    counts_samples = set(counts_df.columns)
    
    # Check consistency
    missing_in_meta = counts_samples - metadata_samples
    missing_in_counts = metadata_samples - counts_samples
    
    if missing_in_meta or missing_in_counts:
        print("\n! Sample inconsistency warning !")
        print(f"Samples in counts missing from metadata: {missing_in_meta}")
        print(f"Samples in metadata missing from counts: {missing_in_counts}")
        
        # Align samples
        common_samples = metadata_samples & counts_samples
        print(f"Common samples: {len(common_samples)}")
        
        # Update counts and metadata
        counts_df = counts_df[list(common_samples)]
        metadata_df = metadata_df[metadata_df['sample_names'].isin(common_samples)]
    
    return metadata_df, counts_df

def main(args):
    """
    Main pipeline function with command-line arguments.
    
    Args:
        args (Namespace): Command-line arguments
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Create count matrix
    print("\n" + "="*50)
    print("Creating count matrix...")
    counts_df = create_counts_df(args.natural_dir, args.samples_dir)
    counts_df.to_csv(os.path.join(args.output_dir, "counts_df.csv"), index=True)
    print("Count matrix saved to counts_df.csv")
    
    # Step 2: Create metadata
    print("\n" + "="*50)
    print("Creating metadata...")
    metadata_df = create_metadata(args.natural_dir, args.samples_dir, counts_df)
    metadata_df.to_csv(os.path.join(args.output_dir, "metadata_df.csv"), index=False)
    print("Metadata saved to metadata_df.csv")
    
    # Step 3: Filter samples by zero ratio
    print("\n" + "="*50)
    print("Filtering samples by zero ratio...")
    filtered_counts, filtered_samples_info = filter_samples_by_zeros(
        counts_df, 
        max_zero_ratio=args.max_zero_ratio
    )
    filtered_samples_info.to_csv(
        os.path.join(args.output_dir, "filtered_samples_info.csv"),
        index=False
    )
    print(f"Filtered {len(filtered_samples_info)} samples")
    
    # Step 4: Update metadata with filtered samples
    valid_samples = filtered_counts.columns.tolist()
    metadata_df = metadata_df[metadata_df['sample_names'].isin(valid_samples)]
    metadata_df, filtered_counts = validate_metadata(metadata_df, filtered_counts)
    
    # Step 5: Filter genes by expression
    print("\n" + "="*50)
    print("Filtering genes by expression...")
    filtered_counts = filter_genes_by_expression(
        filtered_counts,
        metadata_df, 
        min_samples_percent=args.min_expression_percent
    )
    filtered_counts.to_csv(os.path.join(args.output_dir, "counts_df_filtered.csv"), index=True)
    print(f"Filtered genes saved to counts_df_filtered.csv")
    
    # Step 6: Prepare for VST
    if 'sample_names' in metadata_df.columns:
        metadata_df = metadata_df.set_index('sample_names')
    
    # Step 7: Perform VST
    print("\n" + "="*50)
    print("Performing VST transformation...")
    results_counts_vst, counts_df_normed = VST(
        filtered_counts, 
        metadata_df, 
        args.threads, 
        aligned_samples=args.aligned_samples
    )
    
    # Step 8: Save results
    results_counts_vst.to_csv(os.path.join(args.output_dir, "results_counts_vst.csv"), index=True)
    counts_df_normed.to_csv(os.path.join(args.output_dir, "counts_df_normed.csv"), index=True)
    
    print("\n" + "="*50)
    print("Pipeline completed successfully!")
    print(f"VST results saved to results_counts_vst.csv")
    print(f"Normalized counts saved to counts_df_normed.csv")

if __name__ == "__main__":
    # Set up command-line arguments
    parser = argparse.ArgumentParser(
        description="RNA-Seq Analysis Pipeline with VST Transformation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--natural_dir", 
        required=True,
        help="Directory containing natural environment samples"
    )
    parser.add_argument(
        "--samples_dir", 
        required=True,
        help="Directory containing treated/variable samples"
    )
    
    # Optional arguments with defaults
    parser.add_argument(
        "--output_dir", 
        default="output",
        help="Output directory for results"
    )
    parser.add_argument(
        "--threads", 
        type=int, 
        default=8,
        help="Number of CPU threads to use"
    )
    parser.add_argument(
        "--aligned_samples", 
        action="store_true",
        default=True,
        help="Align samples between counts and metadata. If don't chose this in terminal, it will work"
    )
    parser.add_argument(
        "--no_align", 
        dest="aligned_samples",
        action="store_false",
        help="Disable sample alignment. If don't chose this in terminal, it doesn't work"
    )
    parser.add_argument(
        "--max_zero_ratio", 
        type=float, 
        default=0.25,
        help="Maximum zero ratio for sample filtering"
    )
    parser.add_argument(
        "--min_expression_percent", 
        type=float, 
        default=0.95,
        help="Minimum expression percentage for gene filtering"
    )
    
    # Parse arguments and run pipeline
    args = parser.parse_args()
    main(args)