from flask import Flask, request, jsonify, render_template
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io, base64
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

# Global light style for all plots
plt.rcParams.update({
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#ffffff',
    'axes.edgecolor': '#cccccc',
    'axes.labelcolor': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'text.color': '#111111',
    'grid.color': '#e5e5e5',
    'grid.linewidth': 0.8,
    'font.family': 'DejaVu Sans',
    # Speed optimisations
    'path.simplify': True,
    'path.simplify_threshold': 1.0,
    'agg.path.chunksize': 10000,
})
sns.set_theme(style='whitegrid')

df_cache = {}

# Pre-warm matplotlib: load fonts + renderer NOW so the first plot request is instant
def _prewarm():
    _f, _a = plt.subplots(1, 1, figsize=(1, 1))
    _f.savefig(io.BytesIO(), format='png', dpi=36)
    plt.close(_f)

_prewarm()

def fig_to_b64(fig, fmt='jpeg'):
    buf = io.BytesIO()
    # JPEG: 3-5x faster to encode than PNG, 60% smaller payload
    # White background required for JPEG (no transparency support)
    if fmt == 'jpeg':
        fig.savefig(buf, format='jpeg', quality=88, dpi=72,
                    facecolor='white', bbox_inches=None)
    else:
        fig.savefig(buf, format='png', dpi=85, facecolor=fig.get_facecolor())
    buf.seek(0)
    result = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return result, fmt

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400
    try:
        df = pd.read_csv(file)
        df_cache['df'] = df
        num_cols = df.select_dtypes(include='number').columns.tolist()
        cat_cols = df.select_dtypes(exclude='number').columns.tolist()
        stats = {}
        for col in num_cols:
            s = df[col].describe()
            stats[col] = {k: round(float(v), 4) for k, v in s.items()}
        return jsonify({
            'columns': df.columns.tolist(),
            'numeric_columns': num_cols,
            'categorical_columns': cat_cols,
            'rows': len(df),
            'cols_count': len(df.columns),
            'preview': df.head(5).fillna('').astype(str).to_dict(orient='records'),
            'stats': stats,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/plot', methods=['POST'])
def plot():
    df = df_cache.get('df')
    if df is None:
        return jsonify({'error': 'No data loaded. Please upload a CSV first.'}), 400
    
    try:
        body = request.json
        plot_type = body.get('plot_type', 'histogram')
        x_col = body.get('x_col', '')
        y_col = body.get('y_col', '')
        hue_col = body.get('hue_col', '')
        PALETTE = 'mako'

        print(f">>> Request: {plot_type} | X: {x_col} | Y: {y_col} | Hue: {hue_col}")

        # Hue safety check
        hue = None
        if hue_col and hue_col in df.columns:
            if df[hue_col].nunique() > 15:
                return jsonify({'error': f"Too many unique values in '{hue_col}' for coloring (Max 15). Choose a simpler column."}), 400
            hue = hue_col

        # Downsampling limits for performance
        sample_sizes = {
            'heatmap':  50000,
            'pairplot':  1000,
            'joint':     3000,
            'bar':       5000,
            'barh':      5000,
            'scatter':   5000,
            'line':      2000,
            'histogram': 30000,
            'kde':       5000,
            'box':       10000,
            'violin':    2000,
            'count':     30000,
            'strip':     2000,
            'swarm':       300,
            'reg':        2000,
            'residual':   2000,
            'ecdf':      20000,
            'rug':        5000,
            'point':      5000,
            'hexbin':    10000,
            'pie':       30000,
            'area':       2000,
            'step':      10000,
        }
        
        is_downsampled = False
        limit = sample_sizes.get(plot_type, 10000)

        if plot_type == 'heatmap':
            num_df = df.select_dtypes(include='number')
            if num_df.empty:
                return jsonify({'error': 'No numeric columns found for heatmap'}), 400
            
            orig_len = len(num_df)
            if orig_len > limit:
                num_df = num_df.sample(limit, random_state=42)
                is_downsampled = True
                
            fig, ax = plt.subplots(figsize=(max(8, len(num_df.columns)), max(6, len(num_df.columns) - 1)))
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#ffffff')
            sns.heatmap(num_df.corr(), annot=True, fmt='.2f', cmap='coolwarm',
                        linewidths=0.5, linecolor='#ffffff', ax=ax,
                        annot_kws={'size': 9, 'color': '#111111'})
            ax.set_title('Correlation Heatmap', color='#111111', pad=12, fontsize=14)
            plt.tight_layout()
            
            img, fmt = fig_to_b64(fig, fmt='png')  # PNG for heatmap: sharp annotation text
            resp = {'image': img, 'fmt': fmt}
            if is_downsampled:
                resp.update({'downsampled': True, 'sample_size': limit, 'original_size': orig_len})
            return jsonify(resp)

        if plot_type == 'pairplot':
            num_cols = df.select_dtypes(include='number').columns[:5].tolist()
            if not num_cols:
                return jsonify({'error': 'No numeric columns found for pairplot'}), 400
            
            orig_len = len(df)
            pair_df = df
            if orig_len > limit:
                pair_df = df.sample(limit, random_state=42)
                is_downsampled = True
                
            g = sns.pairplot(pair_df[num_cols + ([hue] if hue else [])], hue=hue,
                             diag_kind='kde', plot_kws={'alpha': 0.5},
                             palette=PALETTE)
            g.figure.patch.set_facecolor('#ffffff')
            
            img, fmt = fig_to_b64(g.figure, fmt='png')  # PNG for pairplot: sharp detail
            resp = {'image': img, 'fmt': fmt}
            if is_downsampled:
                resp.update({'downsampled': True, 'sample_size': limit, 'original_size': orig_len})
            return jsonify(resp)

        if plot_type == 'joint':
            if not x_col or not y_col:
                return jsonify({'error': 'Joint plot requires both X and Y columns.'}), 400
            if not pd.api.types.is_numeric_dtype(df[x_col]) or not pd.api.types.is_numeric_dtype(df[y_col]):
                return jsonify({'error': 'Joint plot requires both X and Y to be numeric columns.'}), 400

            orig_len = len(df)
            joint_df = df
            if orig_len > limit:
                joint_df = df.sample(limit, random_state=42)
                is_downsampled = True

            g = sns.jointplot(data=joint_df, x=x_col, y=y_col, hue=hue,
                              kind='scatter', palette=PALETTE,
                              marginal_kws={'fill': True},
                              joint_kws={'alpha': 0.6, 's': 40})
            g.figure.patch.set_facecolor('#ffffff')
            g.ax_joint.set_facecolor('#ffffff')
            g.ax_marg_x.set_facecolor('#ffffff')
            g.ax_marg_y.set_facecolor('#ffffff')
            g.figure.suptitle(f'Joint — {x_col} vs {y_col}',
                              y=1.02, fontsize=14, color='#111111', fontweight='bold')

            img, fmt = fig_to_b64(g.figure, fmt='jpeg')
            resp = {'image': img, 'fmt': fmt}
            if is_downsampled:
                resp.update({'downsampled': True, 'sample_size': limit, 'original_size': orig_len})
            return jsonify(resp)

        # For all other plot types:
        plot_df = df
        orig_len = len(df)
        if orig_len > limit:
            plot_df = df.sample(limit, random_state=42)
            is_downsampled = True
            print(f">>> Downsampled {plot_type} from {orig_len} to {limit} rows")

        # Smaller figure = fewer pixels = faster render
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#ffffff')
        plt.subplots_adjust(left=0.15, right=0.97, top=0.88, bottom=0.20)

        # Helper: only supply palette when hue is active (avoids seaborn FutureWarning)
        pal_kw = {'palette': PALETTE} if hue else {'color': '#6366f1'}

        if plot_type == 'bar':
            data = plot_df[[x_col, y_col]].dropna().head(30) if y_col else plot_df[[x_col]].dropna().head(30)
            sns.barplot(data=data, x=x_col, y=y_col, hue=hue, ax=ax, **pal_kw)
        elif plot_type == 'scatter':
            sns.scatterplot(data=plot_df, x=x_col, y=y_col, hue=hue, ax=ax,
                            alpha=0.6, s=20, rasterized=True, **pal_kw)
        elif plot_type == 'line':
            sns.lineplot(data=plot_df.head(200), x=x_col, y=y_col, hue=hue, ax=ax, **pal_kw)
        elif plot_type == 'histogram':
            sns.histplot(plot_df[x_col].dropna(), kde=False, ax=ax,
                         color='#6366f1', edgecolor='#ffffff')
        elif plot_type == 'kde':
            sns.kdeplot(data=plot_df, x=x_col, hue=hue, fill=True, ax=ax, alpha=0.6, **pal_kw)
        elif plot_type == 'box':
            sns.boxplot(data=plot_df, x=x_col, y=y_col, hue=hue, ax=ax, **pal_kw)
        elif plot_type == 'violin':
            sns.violinplot(data=plot_df, x=x_col, y=y_col, hue=hue, ax=ax, **pal_kw)
        elif plot_type == 'count':
            order = plot_df[x_col].value_counts().index[:15]
            sns.countplot(data=plot_df, x=x_col, hue=hue, order=order, ax=ax, **pal_kw)
        elif plot_type == 'strip':
            sns.stripplot(data=plot_df, x=x_col, y=y_col, hue=hue, ax=ax,
                          alpha=0.5, jitter=True, rasterized=True, size=3, **pal_kw)
        elif plot_type == 'swarm':
            sns.swarmplot(data=plot_df.sample(min(300, len(plot_df))), x=x_col, y=y_col,
                          hue=hue, ax=ax, size=3, **pal_kw)
        elif plot_type == 'reg':
            if not pd.api.types.is_numeric_dtype(plot_df[x_col]) or not pd.api.types.is_numeric_dtype(plot_df[y_col]):
                return jsonify({'error': 'Regression requires both X and Y to be numeric columns.'}), 400
            sns.regplot(data=plot_df, x=x_col, y=y_col, ax=ax, color='#6366f1',
                        scatter_kws={'alpha': 0.6, 's': 40}, line_kws={'color': '#059669'})
        elif plot_type == 'residual':
            if not pd.api.types.is_numeric_dtype(plot_df[x_col]) or not pd.api.types.is_numeric_dtype(plot_df[y_col]):
                return jsonify({'error': 'Residual plot requires both X and Y to be numeric columns.'}), 400
            sns.residplot(data=plot_df, x=x_col, y=y_col, ax=ax, color='#6366f1',
                          scatter_kws={'alpha': 0.6, 's': 40})
        elif plot_type == 'ecdf':
            sns.ecdfplot(data=plot_df, x=x_col, hue=hue, ax=ax, **pal_kw)
        elif plot_type == 'rug':
            sns.rugplot(data=plot_df, x=x_col, hue=hue, ax=ax, height=0.1, alpha=0.5, **pal_kw)
        elif plot_type == 'point':
            sns.pointplot(data=plot_df, x=x_col, y=y_col, hue=hue, ax=ax,
                          capsize=0.1, **pal_kw)
        elif plot_type == 'barh':
            data_bh = plot_df[[x_col, y_col]].dropna().head(30) if y_col else plot_df[[x_col]].dropna().head(30)
            sns.barplot(data=data_bh, x=y_col if y_col else x_col,
                        y=x_col if y_col else None, hue=hue, ax=ax,
                        orient='h', **pal_kw)
        elif plot_type == 'hexbin':
            if not pd.api.types.is_numeric_dtype(plot_df[x_col]) or not pd.api.types.is_numeric_dtype(plot_df[y_col]):
                return jsonify({'error': 'Hexbin requires both X and Y to be numeric columns.'}), 400
            hb = ax.hexbin(plot_df[x_col].dropna(), plot_df[y_col].dropna(),
                           gridsize=30, cmap='Blues', mincnt=1)
            plt.colorbar(hb, ax=ax, label='Count')
        elif plot_type == 'pie':
            counts = plot_df[x_col].value_counts().head(12)
            colors = plt.cm.Set3.colors[:len(counts)]
            wedges, texts, autotexts = ax.pie(
                counts.values, labels=counts.index,
                autopct='%1.1f%%', startangle=140,
                colors=colors, pctdistance=0.82,
                wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
            for t in texts: t.set_fontsize(10); t.set_color('#333333')
            for at in autotexts: at.set_fontsize(9); at.set_color('#111111')
            ax.axis('equal')
        elif plot_type == 'area':
            if not pd.api.types.is_numeric_dtype(plot_df[y_col]):
                return jsonify({'error': 'Area chart requires Y to be a numeric column.'}), 400
            area_df = plot_df[[x_col, y_col]].dropna().head(200)
            ax.fill_between(range(len(area_df)), area_df[y_col].values,
                            alpha=0.4, color='#6366f1', linewidth=0)
            ax.plot(range(len(area_df)), area_df[y_col].values,
                    color='#6366f1', linewidth=1.5)
            ax.set_xticks(range(0, len(area_df), max(1, len(area_df)//10)))
            ax.set_xticklabels(
                area_df[x_col].iloc[::max(1, len(area_df)//10)].astype(str),
                rotation=40, ha='right')
        elif plot_type == 'step':
            if not pd.api.types.is_numeric_dtype(plot_df[y_col]):
                return jsonify({'error': 'Step chart requires Y to be a numeric column.'}), 400
            step_df = plot_df[[x_col, y_col]].dropna().sort_values(x_col).head(300)
            ax.step(range(len(step_df)), step_df[y_col].values,
                    where='mid', color='#6366f1', linewidth=2)
            ax.fill_between(range(len(step_df)), step_df[y_col].values,
                            step='mid', alpha=0.15, color='#6366f1')

        ax.set_title(f'{plot_type.title()} — {x_col}' + (f' vs {y_col}' if y_col else ''),
                     color='#111111', pad=25, fontsize=18, fontweight='bold')
        
        ax.set_xlabel(x_col, fontsize=14, color='#333333', labelpad=15)
        if y_col:
            ax.set_ylabel(y_col, fontsize=14, color='#333333', labelpad=15)

        for spine in ax.spines.values():
            spine.set_edgecolor('#cccccc')
            
        plt.xticks(rotation=40, ha='right', fontsize=11)
        plt.yticks(fontsize=11)
        
        img, fmt = fig_to_b64(fig, fmt='jpeg')
        resp = {'image': img, 'fmt': fmt}
        if is_downsampled:
            resp.update({'downsampled': True, 'sample_size': limit, 'original_size': orig_len})
        return jsonify(resp)

    except Exception as e:
        import traceback
        traceback.print_exc()
        plt.close('all')
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    df = df_cache.get('df')
    if df is None:
        return jsonify({'error': 'No data loaded.'}), 400
    
    try:
        body = request.json
        target = body.get('target')
        features = body.get('features', []) # List of feature names

        if not target or not features:
            return jsonify({'error': 'Please select both Target (Y) and at least one Feature (X).'}), 400

        # Data cleaning: drop NaNs in selected columns
        cols = [target] + features
        clean_df = df[cols].dropna()

        if len(clean_df) < 5:
            return jsonify({'error': 'Not enough data points after dropping missing values (min 5 required).'}), 400

        X = clean_df[features]
        y = clean_df[target]

        # Ensure all are numeric
        if not all(pd.api.types.is_numeric_dtype(clean_df[c]) for c in cols):
            return jsonify({'error': 'Prediction only works with numeric columns.'}), 400

        model = LinearRegression()
        model.fit(X, y)

        r2 = model.score(X, y)
        coefs = model.coef_.tolist()
        intercept = float(model.intercept_)

        # Build equation string
        eq = f"y = {intercept:.4f}"
        for i, f in enumerate(features):
            eq += f" + ({coefs[i]:.4f} * {f})"

        return jsonify({
            'r2': round(r2, 4),
            'intercept': intercept,
            'coefficients': {features[i]: round(c, 4) for i, c in enumerate(coefs)},
            'equation': eq,
            'features_count': len(features),
            'target_range': {
                'min': float(clean_df[target].min()),
                'max': float(clean_df[target].max())
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print('\n>>> EDA Analyzer running at: http://localhost:5000\n')
    app.run(debug=True, port=5000)
