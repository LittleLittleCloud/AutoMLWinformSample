using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;

namespace AutoMLSample;

public partial class Form1 : Form
{
    MLContext mlContext;
    CancellationTokenSource cts;

    public Form1()
    {
        InitializeComponent();
    }

    private void Form1_Load(object sender, EventArgs e)
    {
        button2.Enabled = false;
    }

    private async void button1_Click(object sender, EventArgs e)
    {
        string sampleData = "Sample.csv";
        mlContext = new MLContext();

        // Infer column information
        ColumnInferenceResults columnInference = mlContext.Auto()
                                                        .InferColumns(sampleData,
                                                            labelColumnName: "Souha",
                                                            groupColumns: false);
        // Create text loader
        TextLoader loader = mlContext.Data.CreateTextLoader(columnInference.TextLoaderOptions);

        // Load data into IDataView
        IDataView data = loader.Load(sampleData);

        TrainTestData trainValidationData = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

        SweepablePipeline pipeline = mlContext.Auto()
                            .Featurizer(data, columnInformation: columnInference.ColumnInformation)
                                .Append(mlContext.Auto()
                                    .Regression(labelColumnName: columnInference.ColumnInformation.LabelColumnName));

        AutoMLExperiment experiment = mlContext.Auto().CreateExperiment();

        var regressionMetric = RegressionMetric.RootMeanSquaredError;
        experiment
            .SetPipeline(pipeline)
            .SetRegressionMetric(regressionMetric,
                                    labelColumn: columnInference.ColumnInformation.LabelColumnName)
            .SetTrainingTimeInSeconds(100)  // Training time in sec
            .SetDataset(trainValidationData);

        var monitor = new AutoMLMonitor(pipeline, richTextBox1);
        experiment.SetMonitor(monitor);

        cts = new CancellationTokenSource();
        button1.Enabled = false;
        button2.Enabled = true;
        _ = Task.Run(async () => {
            await experiment.RunAsync(cts.Token);
            button1.Enabled = true;
            button2.Enabled = false;
            richTextBox1.AppendText(Environment.NewLine +"Training Finished!!" + Environment.NewLine);
        });
    }

    private void button2_Click(object sender, EventArgs e)
    {
        cts.Cancel();
    }
}
