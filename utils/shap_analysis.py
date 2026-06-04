import shap
import matplotlib.pyplot as plt
import numpy as np

def generate_shap_plot(model, input_data, feature_names, output_path):
    """
    Generate a SHAP waterfall plot for a single prediction.
    """
    explainer = shap.Explainer(model)
    # Convert input list to 2D numpy array
    input_array = np.array([input_data])
    
    # Calculate SHAP values
    shap_values = explainer(input_array)
    
    # Generate the waterfall plot for the first (and only) instance
    plt.figure()
    
    # Create an Explanation object suitable for waterfall plot
    exp = shap.Explanation(
        values=shap_values.values[0], 
        base_values=shap_values.base_values[0], 
        data=input_array[0], 
        feature_names=feature_names
    )
    
    shap.plots.waterfall(exp, show=False)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
