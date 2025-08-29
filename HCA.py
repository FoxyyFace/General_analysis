import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.gridspec as gridspec
import os
import sys
import argparse
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference

def load_data(csv_path):
    """加载CSV文件"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"文件不存在: {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)
    return df

def plot_dendrogram(model, labels, ax, orientation='left', leaf_font_size=8, leaf_rotation=0):
    """
    绘制树状图（改进版）
    """
    # 创建链接矩阵
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # 叶子节点
            else:
                current_count += counts[child_idx - n_samples]  # 合并节点
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, 
                                    model.distances_,
                                    counts]).astype(float)
    
    # 绘制树状图
    ddata = dendrogram(linkage_matrix, 
                      ax=ax,
                      orientation=orientation,
                      labels=labels,
                      leaf_font_size=leaf_font_size,
                      leaf_rotation=leaf_rotation,
                      color_threshold=0,
                      above_threshold_color="#33B1FF",
                      no_labels=False)
    
    if orientation == 'left':
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    
    return ddata

def apply_vst(df_numeric):
    """
    使用pyDESeq2进行VST标准化
    """
    try:
        # 创建伪元数据
        metadata = pd.DataFrame(index=df_numeric.columns)
        metadata['condition'] = 'A'  # 创建伪条件列
        
        inference = DefaultInference(n_cpus=1)    
        dds = DeseqDataSet(
            counts=df_numeric.T,  # 转置为样本×基因
            metadata=metadata,
            design="~condition",  
            refit_cooks=True,
            inference=inference,
        )
        dds.fit_size_factors()
        dds.fit_genewise_dispersions()
        dds.fit_dispersion_trend()
        dds.fit_dispersion_prior()
        dds.vst()  # 执行VST转换

        # 获取VST转换后的数据
        vst_df = pd.DataFrame(
            dds.layers["vst_counts"],
            index=dds.obs_names,  # 样本名称
            columns=dds.var_names  # 特征名称
        ).T  # 转置回特征×样本
        
        return vst_df
    
    except Exception as e:
        print(f"VST处理出错: {e}")
        raise

def calculate_feature_variance(df_normalized):
    """
    计算标准化后每个特征(Chimera)的方差并排序
    """
    # 计算每行(特征)的方差
    variances = df_normalized.var(axis=1)
    
    # 创建DataFrame保存结果
    variance_df = pd.DataFrame({
        'Feature': variances.index,
        'Variance': variances.values
    })
    
    # 按方差降序排列
    variance_df = variance_df.sort_values('Variance', ascending=False)
    
    return variance_df

def normalize_features(df_numeric, method='zscore'):
    """
    对特征进行标准化（行方向）
    """
    if method.lower() == 'vst':
        return apply_vst(df_numeric)
    elif method.lower() == 'zscore':
        scaler = StandardScaler()
        data_standardized = scaler.fit_transform(df_numeric)
        return pd.DataFrame(data_standardized, index=df_numeric.index, columns=df_numeric.columns)
    else:
        raise ValueError(f"未知的标准化方法: {method}")

def normalize_heatmap_data(df, method='row'):
    """
    对热图数据进行归一化处理
    """
    if method == 'row':
        # 行归一化：每行独立归一化到[-1,1]范围
        normalized_df = df.copy()
        for i in range(df.shape[0]):
            row = df.iloc[i].values
            max_val = np.max(np.abs(row))
            if max_val > 0:
                normalized_df.iloc[i] = row / max_val
        return normalized_df
    elif method == 'global':
        # 全局归一化：整个数据集归一化到[-1,1]范围
        max_val = np.max(np.abs(df.values))
        return df / max_val
    else:
        return df

def plot_clustered_heatmap(df, output_image="clustered_heatmap.png", 
                           standardization='zscore', top_n=None,
                           output_prefix="cluster", heatmap_normalization='row'):
    """
    使用层次聚类绘制热图（含树状图）
    """
    # 数据预处理
    df_numeric = df.select_dtypes(include=[np.number])
    if df_numeric.empty:
        raise ValueError("数据中没有数值列")
    
    # 应用标准化
    df_normalized = normalize_features(df_numeric, method=standardization)
    
    # 计算特征方差并保存（使用标准化后的数据）
    variance_df = calculate_feature_variance(df_normalized)
    variance_df.to_csv(f"{output_prefix}_feature_variance.csv", index=False)
    print(f"特征方差数据已保存至: {output_prefix}_feature_variance.csv")
    
    # 特征选择：按标准化后的方差选择top特征
    if top_n:
        print(f"选择方差最大的 {top_n} 个特征")
        # 获取top特征名称
        top_features = variance_df.head(top_n)['Feature'].tolist()
        # 筛选标准化后的数据
        df_selected = df_normalized.loc[top_features]
        print(f"选择后数据形状: {df_selected.shape}")
    else:
        df_selected = df_normalized
    
    print("正在进行层次聚类...")
    
    # 对行（特征）进行层次聚类
    row_cluster = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0,  # 设置适当的距离阈值
        metric='euclidean',
        linkage='average',
        compute_distances=True
    )
    
    # 对列（样本）进行层次聚类
    col_cluster = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0,
        metric='euclidean',
        linkage='average',
        compute_distances=True
    )
    
    # 拟合聚类模型
    row_labels = row_cluster.fit_predict(df_selected)
    col_labels = col_cluster.fit_predict(df_selected.T)
    
    # 获取聚类顺序
    row_order = get_cluster_order(row_cluster, df_selected.index)
    col_order = get_cluster_order(col_cluster, df_selected.columns)
    
    # 按聚类顺序重新排列数据
    df_clustered = df_selected.iloc[row_order, col_order]
    
    # 输出聚类后的数值矩阵
    matrix_output = f"{output_prefix}_clustered_data.csv"
    df_clustered.to_csv(matrix_output)
    print(f"聚类后的热图数值矩阵已保存至: {matrix_output}")
    
    # 对热图数据进行归一化处理
    df_heatmap = normalize_heatmap_data(df_clustered, method=heatmap_normalization)
    
    # 动态计算图像高度
    num_features = df_clustered.shape[0]
    num_samples = df_clustered.shape[1]
    
    # 基础高度（针对样本树状图和标签空间）
    base_height = 6
    
    # 特征数越多，高度越大（0.2英寸/行）
    height_per_feature = 0.2
    fig_height = max(12, base_height + num_features * height_per_feature)
    
    # 宽度固定（适应标签）
    fig_width = min(24, max(12, num_samples * 0.2))
    
    # 动态调整行树状图宽度比例
    # 特征越多，行树状图越宽
    dendro_width_ratio = min(4.0, max(2.0, 0.2 + num_features * 0.001))
    heatmap_width_ratio = 4
    label_width_ratio = 0.3
    
    # 确保总比例为1
    total_width = dendro_width_ratio + heatmap_width_ratio + label_width_ratio
    dendro_width_ratio /= total_width
    heatmap_width_ratio /= total_width
    label_width_ratio /= total_width
    
    print(f"行树状图宽度比例: {dendro_width_ratio:.2f}, 热图宽度比例: {heatmap_width_ratio:.2f}, 标签宽度比例: {label_width_ratio:.2f}")
    
    # 设置网格布局
    gs = gridspec.GridSpec(2, 3, 
                          width_ratios=[dendro_width_ratio, heatmap_width_ratio, label_width_ratio],
                          height_ratios=[1, 4],
                          wspace=0.02, hspace=0.02)
    
    # 创建画布（动态高度）
    fig = plt.figure(figsize=(fig_width, fig_height))
    print(f"热图尺寸: 宽={fig_width:.1f}英寸, 高={fig_height:.1f}英寸")
    
    # 绘制行（特征）树状图（左侧）- 动态宽度
    ax_row_dendro = fig.add_subplot(gs[1, 0])
    # 增加标签字体大小和旋转角度
    plot_dendrogram(row_cluster, df_selected.index[row_order], 
                   ax=ax_row_dendro, orientation='left',
                   leaf_font_size=10, leaf_rotation=0)
    ax_row_dendro.set_title('Feature Clustering', fontsize=12, pad=10)
    
    # 绘制列（样本）树状图（顶部）
    ax_col_dendro = fig.add_subplot(gs[0, 1])
    plot_dendrogram(col_cluster, df_selected.columns[col_order], 
                   ax=ax_col_dendro, orientation='top',
                   leaf_font_size=8)
    ax_col_dendro.set_title('Sample Clustering', fontsize=12, pad=10)
    
    # 绘制热图（主图）
    ax_heatmap = fig.add_subplot(gs[1, 1])
    
    # 确定色图范围（归一化后范围固定为[-1,1]）
    vmin, vmax = -1, 1
    
    im = ax_heatmap.imshow(df_heatmap.values, 
                         aspect='auto',
                         cmap='viridis',
                         interpolation='nearest',
                         vmin=vmin, vmax=vmax)
    
    # 设置列标签（样本，底部）
    ax_heatmap.set_xticks(range(len(col_order)))
    ax_heatmap.set_xticklabels([df_clustered.columns[i] for i in range(len(col_order))], 
                              rotation=90, fontsize=8)
    
    # 移除热图左侧的y轴刻度
    ax_heatmap.set_yticks([])
    
    # 添加右侧的行标签（特征）
    ax_labels = fig.add_subplot(gs[1, 2])
    ax_labels.set_yticks(range(len(row_order)))
    ax_labels.set_yticklabels([df_clustered.index[i] for i in range(len(row_order))], 
                             fontsize=8, ha='left', va='center')
    ax_labels.yaxis.set_tick_params(left=False, labelleft=False, right=True, labelright=True)
    ax_labels.set_xticks([])
    ax_labels.set_frame_on(False)
    
    # 添加颜色条
    cax = fig.add_axes([0.93, 0.9, 0.02, 0.1])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Normalized Value', fontsize=10)
    
    # 设置主标题
    title = f'Hierarchical Clustering Heatmap\n({standardization} standardization'
    if top_n:
        title += f', top {top_n} features by variance'
    title += ')'
    
    plt.suptitle(title, fontsize=16, y=0.95)
    
    # 保存和显示
    # 根据输出文件类型保存
    if output_image.lower().endswith('.pdf'):
        # 对于PDF，使用矢量格式保存
        plt.savefig(output_image, bbox_inches='tight')
    else:
        # 对于其他格式，使用高DPI保存
        plt.savefig(output_image, dpi=300, bbox_inches='tight')
    
    print(f"热图已保存至: {output_image}")
    plt.close()  # 避免在非交互式环境中显示图像
    
    # 解释VST值的含义（如果需要）
    if standardization == 'vst':
        print("\n### VST标准化数值说明 ###")
        print("VST转换后的值反映了Chimera的相对表达水平：")
        print(" - 负值: 低于该Chimera在所有样本中的平均表达水平")
        print(" - 正值: 高于该Chimera在所有样本中的平均表达水平")
        print(" - 数值大小: 表示偏离平均表达水平的程度")
        print(" - 零值: 等于平均表达水平")
        print("示例:")
        print(f"  值=-1.5: 表示表达水平显著低于平均")
        print(f"  值=+1.2: 表示表达水平高于平均")
        print("注意：不同Chimera之间的绝对值不能直接比较")
    
    return df_clustered, row_cluster, col_cluster

def get_cluster_order(model, labels):
    """
    获取聚类顺序（基于树状图叶子节点顺序）
    """
    # 创建链接矩阵
    linkage_matrix = np.column_stack([model.children_, 
                                    model.distances_,
                                    np.zeros(model.children_.shape[0])]).astype(float)
    
    # 获取叶子节点顺序
    d = dendrogram(linkage_matrix, no_plot=True)
    return d['leaves']

def extract_cluster_results(df, row_cluster, col_cluster, output_prefix="cluster"):
    """
    提取聚类结果
    """
    # 提取特征聚类结果
    feature_clusters = {}
    for i, label in enumerate(row_cluster.labels_):
        if label not in feature_clusters:
            feature_clusters[label] = []
        feature_clusters[label].append(df.index[i])
    
    # 按聚类大小排序
    sorted_feature_clusters = sorted(feature_clusters.items(), key=lambda x: len(x[1]), reverse=True)
    
    with open(f"{output_prefix}_features.txt", "w") as f:
        for cluster_id, (label, items) in enumerate(sorted_feature_clusters):
            f.write(f"Feature Cluster {cluster_id} (label={label}, size={len(items)}):\n")
            for item in items:
                f.write(f"{item}\n")
            f.write("\n")
    
    # 提取样本聚类结果
    sample_clusters = {}
    for i, label in enumerate(col_cluster.labels_):
        if label not in sample_clusters:
            sample_clusters[label] = []
        sample_clusters[label].append(df.columns[i])
    
    sorted_sample_clusters = sorted(sample_clusters.items(), key=lambda x: len(x[1]), reverse=True)
    
    with open(f"{output_prefix}_samples.txt", "w") as f:
        for cluster_id, (label, items) in enumerate(sorted_sample_clusters):
            f.write(f"Sample Cluster {cluster_id} (label={label}, size={len(items)}):\n")
            for item in items:
                f.write(f"{item}\n")
            f.write("\n")
    
    print(f"聚类结果已保存至: {output_prefix}_features.txt 和 {output_prefix}_samples.txt")
    
    # 显示统计信息
    print(f"\n特征聚类: {len(sorted_feature_clusters)} 个聚类")
    for cluster_id, (label, items) in enumerate(sorted_feature_clusters[:5]):
        print(f"  聚类 {cluster_id} (label={label}): {len(items)} 个特征")
    
    print(f"\n样本聚类: {len(sorted_sample_clusters)} 个聚类")
    for cluster_id, (label, items) in enumerate(sorted_sample_clusters[:5]):
        print(f"  聚类 {cluster_id} (label={label}): {len(items)} 个样本")

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='层次聚类分析')
    parser.add_argument('input_file', help='输入CSV文件路径')
    parser.add_argument('-o', '--output', default='clustered_heatmap.pdf', 
                       help='输出图像文件名（推荐PDF格式，默认为clustered_heatmap.pdf）')
    parser.add_argument('-p', '--prefix', default='cluster', 
                       help='输出文件前缀')
    parser.add_argument('-s', '--standardization', default='vst', 
                       choices=['zscore', 'vst'],
                       help='标准化方法: zscore 或 vst(默认)')
    parser.add_argument('-t', '--top', type=int, 
                       help='选择方差最大的前N个特征进行聚类')
    parser.add_argument('--extract', action='store_true',
                       help='是否提取聚类结果')
    parser.add_argument('--heatmap-norm', default='row', choices=['row', 'global', 'none'],
                       help='热图数据归一化方法: row(每行独立归一化,默认), global(全局归一化), none(不归一化)')
    
    args = parser.parse_args()
    
    try:
        # 加载数据
        df = load_data(args.input_file)
        
        print(f"数据加载成功，形状: {df.shape}")
        print(f"原始特征数: {df.shape[0]}, 样本数: {df.shape[1]}")
        print(f"数据范围: {df.select_dtypes(include=[np.number]).values.min():.2f} 到 "
              f"{df.select_dtypes(include=[np.number]).values.max():.2f}")
        
        # 绘制热图
        df_clustered, row_cluster, col_cluster = plot_clustered_heatmap(
            df, 
            output_image=args.output,
            standardization=args.standardization,
            top_n=args.top,
            output_prefix=args.prefix,
            heatmap_normalization=args.heatmap_norm
        )
        
        # 提取聚类结果
        if args.extract:
            extract_cluster_results(df_clustered, row_cluster, col_cluster, output_prefix=args.prefix)
            
    except Exception as e:
        print(f"程序运行错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()