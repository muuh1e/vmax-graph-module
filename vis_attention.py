import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_attention_map():
    # 1. Mock Data (matching your successful test shapes)
    # Shape: (8 Latents, 10 Road Elements)
    # We average over the 4 heads for a clearer view
    cnt_latents = 8
    cnt_elements = 10
    
    # Create a "fake" attention matrix where the agent focuses heavily on Element 3 and 7
    attention_matrix = np.random.rand(cnt_latents, cnt_elements) * 0.1
    attention_matrix[:, 3] += 0.8  # High attention on Car #3
    attention_matrix[:, 7] += 0.5  # Medium attention on Road Line #7
    
    # Normalize so rows sum to 1 (like real Softmax)
    attention_matrix = attention_matrix / attention_matrix.sum(axis=1, keepdims=True)

    # 2. Plotting
    plt.figure(figsize=(10, 6))
    
    # Use Heatmap to show intensity
    sns.heatmap(attention_matrix, cmap="viridis", annot=True, fmt=".2f", cbar_kws={'label': 'Attention Score'})
    
    plt.title("Explainability Module: Agent Attention Distribution")
    plt.xlabel("Environment Elements (Cars, Road Lines)")
    plt.ylabel("Agent Latent Queries")
    
    # Add labels for the demo
    plt.xticks(ticks=np.arange(10)+0.5, labels=[f"Obj {i}" for i in range(10)])
    plt.yticks(ticks=np.arange(8)+0.5, labels=[f"Query {i}" for i in range(8)], rotation=0)
    
    print("✅ Generated 'attention_map.png'")
    plt.savefig("attention_map.png")
    plt.show()

if __name__ == "__main__":
    plot_attention_map()