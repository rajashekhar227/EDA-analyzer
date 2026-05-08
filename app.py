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

# Global dark style for all plots
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#8b949e',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'text.color': '#c9d1d9',
    'grid.color': '#21262d',
    'grid.linewidth': 0.8,
    'font.family': 'DejaVu Sans',
})
sns.set_theme(style='darkgrid')

df_cache = {}

def fig_to_b64(fig):
    buf = io.BytesIO()
    # Use high DPI but fixed size to avoid unpredictable cropping from 'tight' logic
    fig.savefig(buf, format='png', dpi=120, facecolor=fig.get_facecolor())
    buf.seek(0)
    result = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return result

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

        if plot_type == 'heatmap':
            num_df = df.select_dtypes(include='number')
            if num_df.empty:
                return jsonify({'error': 'No numeric columns found for heatmap'}), 400
            fig, ax = plt.subplots(figsize=(max(8, len(num_df.columns)), max(6, len(num_df.columns) - 1)))
            fig.patch.set_facecolor('#0d1117')
            ax.set_facecolor('#161b22')
            sns.heatmap(num_df.corr(), annot=True, fmt='.2f', cmap='coolwarm',
                        linewidths=0.5, linecolor='#0d1117', ax=ax,
                        annot_kws={'size': 9, 'color': '#c9d1d9'})
            ax.set_title('Correlation Heatmap', color='#c9d1d9', pad=12, fontsize=14)
            plt.tight_layout()
            return jsonify({'image': fig_to_b64(fig)})

        if plot_type == 'pairplot':
            num_cols = df.select_dtypes(include='number').columns[:5].tolist()
            if not num_cols:
                return jsonify({'error': 'No numeric columns found for pairplot'}), 400
            g = sns.pairplot(df[num_cols + ([hue] if hue else [])], hue=hue,
                             diag_kind='kde', plot_kws={'alpha': 0.5},
                             palette=PALETTE)
            g.figure.patch.set_facecolor('#0d1117')
            return jsonify({'image': fig_to_b64(g.figure)})

        # Large fixed size for better spacing
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#161b22')
        
        # Adjust layout manually to ensure huge room for labels
        # left=0.20, bottom=0.25 gives lots of space for axis labels
        plt.subplots_adjust(left=0.20, right=0.95, top=0.88, bottom=0.25)

        if plot_type == 'bar':
            data = df[[x_col, y_col]].dropna().head(30) if y_col else df[[x_col]].dropna().head(30)
            sns.barplot(data=data, x=x_col, y=y_col, hue=hue, ax=ax, palette=PALETTE)
        elif plot_type == 'scatter':
            sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, ax=ax, palette=PALETTE, alpha=0.7, s=60)
        elif plot_type == 'line':
            sns.lineplot(data=df.head(200), x=x_col, y=y_col, hue=hue, ax=ax, palette=PALETTE)
        elif plot_type == 'histogram':
            sns.histplot(df[x_col].dropna(), kde=True, ax=ax, color='#818cf8', edgecolor='#0d1117')
        elif plot_type == 'kde':
            sns.kdeplot(data=df, x=x_col, hue=hue, fill=True, ax=ax, palette=PALETTE, alpha=0.6)
        elif plot_type == 'box':
            sns.boxplot(data=df, x=x_col, y=y_col, hue=hue, ax=ax, palette=PALETTE)
        elif plot_type == 'violin':
            sns.violinplot(data=df, x=x_col, y=y_col, hue=hue, ax=ax, palette=PALETTE)
        elif plot_type == 'count':
            order = df[x_col].value_counts().index[:15]
            sns.countplot(data=df, x=x_col, hue=hue, order=order, ax=ax, palette=PALETTE)
        elif plot_type == 'strip':
            sns.stripplot(data=df, x=x_col, y=y_col, hue=hue, ax=ax, palette=PALETTE, alpha=0.6, jitter=True)
        elif plot_type == 'swarm':
            sns.swarmplot(data=df.sample(min(500, len(df))), x=x_col, y=y_col, hue=hue, ax=ax, palette=PALETTE)
        elif plot_type == 'reg':
            if not pd.api.types.is_numeric_dtype(df[x_col]) or not pd.api.types.is_numeric_dtype(df[y_col]):
                return jsonify({'error': 'Regression requires both X and Y to be numeric columns.'}), 400
            sns.regplot(data=df, x=x_col, y=y_col, ax=ax, color='#818cf8',
                        scatter_kws={'alpha': 0.5, 's': 40}, line_kws={'color': '#34d399'})

        ax.set_title(f'{plot_type.title()} — {x_col}' + (f' vs {y_col}' if y_col else ''),
                     color='#c9d1d9', pad=25, fontsize=18, fontweight='bold')
        
        ax.set_xlabel(x_col, fontsize=14, color='#8b949e', labelpad=15)
        if y_col:
            ax.set_ylabel(y_col, fontsize=14, color='#8b949e', labelpad=15)

        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')
            
        plt.xticks(rotation=40, ha='right', fontsize=11)
        plt.yticks(fontsize=11)
        
        return jsonify({'image': fig_to_b64(fig)})

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
