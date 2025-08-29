import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
import os
import sys
import logging
import argparse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def load_data(vst_file, metadata_file):
    """加载VST转换后的数据和元数据"""
    logger.info(f"从 {vst_file} 加载VST数据")
    vst_data = pd.read_csv(vst_file, index_col=0)
    
    logger.info(f"从 {metadata_file} 加载元数据")
    metadata = pd.read_csv(metadata_file)
    
    # 检查数据完整性
    logger.info(f"VST数据形状: {vst_data.shape[1]} 个样本, {vst_data.shape[0]} 个基因")
    logger.info(f"元数据形状: {len(metadata)} 个样本")
    
    # 确保没有NaN值
    if vst_data.isnull().any().any():
        logger.warning("在VST数据中发现NaN值 - 用0填充")
        vst_data = vst_data.fillna(0)
    
    # 确保没有无限大值
    if np.isinf(vst_data.values).any():
        logger.warning("在VST数据中发现Inf值 - 用最大值替换")
        max_val = vst_data.replace([np.inf, -np.inf], np.nan).max().max()
        vst_data = vst_data.replace([np.inf, -np.inf], max_val)
    
    return vst_data, metadata

def prepare_groups(metadata):
    """准备分组信息：nature为一组，samples按group分组"""
    logger.info("准备分组信息")
    
    # 确保必需的列存在
    for col in ['condition', 'group']:
        if col not in metadata.columns:
            raise ValueError(f"元数据缺少必需的列: {col}")
    
    # 创建新的分组列
    metadata['pca_group'] = metadata.apply(
        lambda row: 'nature' if row['condition'] == 'nature' else row['group'],
        axis=1
    )
    
    # 记录分组统计
    group_counts = metadata['pca_group'].value_counts()
    logger.info("样本分组统计:")
    for group, count in group_counts.items():
        logger.info(f"  {group}: {count} 个样本")
    
    return metadata

def select_top_variable_genes(vst_data, n_genes=500):
    """选择表达变异最高的基因"""
    # 计算基因的方差
    gene_vars = vst_data.var(axis=1)
    
    # 选择方差最大的基因
    top_genes = gene_vars.sort_values(ascending=False).head(n_genes).index
    top_genes_data = vst_data.loc[top_genes]
    
    logger.info(f"选择表达变异最高的 {len(top_genes)} 个基因")
    
    return top_genes_data

def perform_pca(data, n_components=2, scale=False):
    """执行PCA分析并处理NaN值"""
    # 确保输入数据是DataFrame
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    # 处理可能的数值问题
    data = data.replace([np.inf, -np.inf], np.nan)
    
    # 转置数据：样本在行，基因在列
    data_t = data.T
    
    # 删除包含NaN的样本
    if data_t.isnull().any().any():
        nan_samples = data_t.isnull().any(axis=1).sum()
        logger.warning(f"删除包含NaN的 {nan_samples} 个样本")
        data_t = data_t.dropna(axis=0)
    
    # 保存样本索引用于匹配元数据
    sample_index = data_t.index
    
    # 记录样本数量
    logger.info(f"用于PCA分析的样本数量: {data_t.shape[0]}")
    
    # 执行标准化（如果需要）
    if scale:
        logger.info("使用RobustScaler缩放数据")
        scaler = RobustScaler(quantile_range=(25, 75))
        scaled_data = scaler.fit_transform(data_t)
    else:
        logger.info("不缩放数据")
        scaled_data = data_t.values
    
    # 执行PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)
    
    logger.info(f"PCA分析完成，保留 {n_components} 个主成分")
    
    return pca_result, pca, sample_index

def plot_pca_results(pca_result, metadata, output_dir):
    """绘制PCA结果"""
    # 确保元数据包含必要的列
    required_cols = ['sample_names', 'pca_group', 'condition']
    missing_cols = [col for col in required_cols if col not in metadata.columns]
    if missing_cols:
        raise ValueError(f"元数据缺少必要的列: {', '.join(missing_cols)}")
    
    # 创建结果DataFrame
    pca_df = pd.DataFrame(
        data=pca_result[:, :2],
        columns=['PC1', 'PC2'],
        index=metadata['sample_names']  # 使用样本名称作为索引
    )
    
    # 将分组信息添加到结果DataFrame
    pca_df['group'] = metadata.set_index('sample_names')['pca_group']
    pca_df['condition'] = metadata.set_index('sample_names')['condition']
    
    # 设置图形大小
    plt.figure(figsize=(12, 8))
    
    # 获取唯一的组和颜色
    unique_groups = pca_df['group'].unique()
    palette = sns.color_palette("husl", len(unique_groups))
    color_dict = dict(zip(unique_groups, palette))
    
    # 为nature组添加特殊标记
    nature_mask = pca_df['condition'] == 'nature'
    
    # 绘制散点图
    for group in unique_groups:
        group_mask = pca_df['group'] == group
        group_data = pca_df[group_mask]
        
        # 如果是nature组，添加特殊轮廓
        if group == 'nature':
            plt.scatter(
                group_data['PC1'], 
                group_data['PC2'],
                s=150,
                alpha=0.8,
                color=color_dict[group],
                edgecolor='black',
                linewidths=1.5,
                label=group
            )
        else:
            plt.scatter(
                group_data['PC1'], 
                group_data['PC2'],
                s=100,
                alpha=0.8,
                color=color_dict[group],
                label=group
            )
    
    # 添加图例和标题
    plt.title('PCA Analysis of Gene Expression Data', fontsize=16)
    plt.xlabel('PC1', fontsize=14)
    plt.ylabel('PC2', fontsize=14)
    plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 保存图形
    output_file = os.path.join(output_dir, 'pca_plot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"PCA图保存至: {output_file}")
    
    return pca_df

def save_pca_results(pca_df, pca, output_dir):
    """保存PCA结果和解释方差"""
    # 保存PCA坐标
    pca_coords_file = os.path.join(output_dir, 'pca_coordinates.csv')
    pca_df.to_csv(pca_coords_file)
    logger.info(f"PCA坐标保存至: {pca_coords_file}")
    
    # 保存解释方差
    variance_df = pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
        'explained_variance': pca.explained_variance_,
        'explained_variance_ratio': pca.explained_variance_ratio_
    })
    
    variance_file = os.path.join(output_dir, 'pca_variance.csv')
    variance_df.to_csv(variance_file, index=False)
    logger.info(f"PCA解释方差保存至: {variance_file}")
    
    # 输出解释方差摘要
    logger.info("\n解释方差比例:")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        logger.info(f"PC{i+1}: {ratio:.2%}")
    
    return variance_df

def run_pca_analysis(args):
    """运行完整的PCA分析流程"""
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"开始PCA分析，结果将保存在: {args.output_dir}")
    
    # 输入文件
    vst_file = os.path.join(args.input_dir, args.vst_file)
    metadata_file = os.path.join(args.input_dir, args.metadata_file)
    
    # 1. 加载数据
    vst_data, metadata = load_data(vst_file, metadata_file)
    
    # 2. 准备分组信息
    metadata = prepare_groups(metadata)
    
    # 3. 选择高变异基因
    top_genes_data = select_top_variable_genes(vst_data, n_genes=args.n_genes)
    
    # 4. 执行PCA分析
    pca_result, pca, sample_index = perform_pca(top_genes_data, scale=args.scale_data)
    
    # 5. 筛选与PCA结果匹配的元数据
    valid_metadata = metadata[metadata['sample_names'].isin(sample_index)]
    logger.info(f"用于绘图的样本数量: {len(valid_metadata)}")
    
    # 6. 绘制和保存结果
    pca_df = plot_pca_results(pca_result, valid_metadata, args.output_dir)
    
    # 7. 保存PCA结果
    variance_df = save_pca_results(pca_df, pca, args.output_dir)
    
    # 返回关键结果
    return {
        'pca_coords': pca_df,
        'variance': variance_df,
        'valid_samples': list(sample_index)
    }

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='RNA-Seq数据PCA分析工具',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 输入输出参数
    parser.add_argument('--input_dir', type=str, default='output',
                        help='输入文件目录，包含VST数据和元数据')
    parser.add_argument('--output_dir', type=str, default='pca_results',
                        help='输出结果目录')
    
    # 文件参数
    parser.add_argument('--vst_file', type=str, default='results_counts_vst.csv',
                        help='VST转换后的数据文件名')
    parser.add_argument('--metadata_file', type=str, default='metadata_df.csv',
                        help='元数据文件名')
    
    # 分析参数
    parser.add_argument('--n_genes', type=int, default=500,
                        help='用于PCA分析的高变异基因数量')
    parser.add_argument('--scale_data', action='store_true',
                        help='是否在PCA前缩放数据')
    parser.add_argument('--n_components', type=int, default=2,
                        help='PCA分析保留的主成分数量')
    
    # 日志参数
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='日志记录级别')
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置日志级别
    logger.setLevel(args.log_level)
    
    # 运行分析
    logger.info("=" * 60)
    logger.info("开始PCA分析")
    logger.info(f"输入目录: {args.input_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"VST文件: {args.vst_file}")
    logger.info(f"元数据文件: {args.metadata_file}")
    logger.info(f"高变异基因数量: {args.n_genes}")
    logger.info(f"缩放数据: {'是' if args.scale_data else '否'}")
    logger.info(f"主成分数量: {args.n_components}")
    logger.info("=" * 60)
    
    results = run_pca_analysis(args)
    
    logger.info("PCA分析完成!")

if __name__ == "__main__":
    main()