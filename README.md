# NVIDIA Stock Price Forecasting with ARIMA

This project aims to forecast the stock prices of NVIDIA Corporation using the Autoregressive Integrated Moving Average (ARIMA) model. ARIMA is a popular time series forecasting technique that captures the autoregressive, integrated, and moving average components of a time series.

## Project Overview

The project utilizes historical stock price data of NVIDIA to train an ARIMA model and generate future price predictions. By analyzing the performance of the model using various evaluation metrics, this project provides insights into the effectiveness of the ARIMA model for stock price forecasting and identifies potential areas for improvement.

## Dataset

The project uses a dataset containing historical stock price data of NVIDIA Corporation. The dataset includes the following features:

- Date
- Open price
- High price
- Low price
- Close price
- Adjusted close price
- Volume

## Approach

The project follows these steps:

1. **Data Preprocessing**: Load and preprocess the stock price data, handling any missing values or inconsistencies.
2. **Data Splitting**: Split the dataset into training and testing sets to evaluate the model's performance.
3. **Model Selection**: Determine the appropriate order of the ARIMA model (p, d, q) based on the characteristics of the data.
4. **Model Training**: Train the ARIMA model using the training dataset.
5. **Forecasting**: Use the trained model to generate future stock price predictions on the testing dataset.
6. **Evaluation**: Evaluate the model's performance using various metrics, such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared.
7. **Visualization**: Visualize the actual and predicted stock prices to assess the model's accuracy visually.

## Key Findings

The project evaluates the performance of the ARIMA model using the following metrics:

- **Mean Absolute Error (MAE)**: The MAE for the model is 92.25, indicating that, on average, the model's predictions are off by approximately 92.25 units.
- **Mean Squared Error (MSE)**: The MSE for the model is 13144.37, which is a relatively high score, suggesting poor model performance.
- **R-squared**: The R-squared value for the model is -0.036, which is negative, suggesting that the model is not explaining much of the variability in the data and might be worse than a simple mean-based model.
- **Residual Plots**: The residual plots indicate that the predicted values are a straight horizontal line on the average ending point, suggesting that the residuals are dependent on the average difference and that the model cannot find a pattern in the complex data.

## Conclusion and Future Improvements

The model's performance, as indicated by the given metrics, is not satisfactory. The high MAE and MSE values, along with the negative R-squared, suggest that the model is not providing a meaningful fit to the data.

To improve the model's performance, the following steps can be considered:

1. **Optimize Order Selection**: Implement an Auto ARIMA function to automatically explore multiple orders and select the best one for the given data.
2. **Explore Alternative Models**: Investigate the use of alternative models, such as Seasonal ARIMA (SARIMA) or Long Short-Term Memory (LSTM) models, which may be better suited for complex and non-cyclical data.
3. **Feature Engineering**: Explore feature selection and engineering techniques to identify and incorporate relevant features that might improve the model's performance.
4. **Hyperparameter Tuning**: Perform hyperparameter tuning to find the optimal set of parameters for the chosen model.

## Usage

To replicate or extend this analysis, follow these steps:

1. Clone the repository or download the project files.
2. Install the required dependencies and libraries (e.g., pandas, numpy, scikit-learn, statsmodels).
3. Obtain the NVIDIA stock price dataset and place it in the appropriate directory.
4. Run the Jupyter Notebooks or Python scripts to execute the analysis pipeline.

## Contributing

Contributions to this project are welcome. If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
