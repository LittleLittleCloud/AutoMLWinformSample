using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.AutoML;

namespace AutoMLSample;

// This is from https://learn.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/how-to-use-the-automl-api
// Since it's for console app, I modified to work as windows form app.
internal class AutoMLMonitor : IMonitor
{
    private readonly SweepablePipeline _pipeline;

    // added to show log to RichTextBox
    private readonly RichTextBox _richTextBox;

    // added parameter RichTextBox for logging
    public AutoMLMonitor(SweepablePipeline pipeline, RichTextBox rtbLog)
    {
        _pipeline = pipeline;
        // added
        _richTextBox = rtbLog;
    }

    // Since it's not required.
    // public IEnumerable<TrialResult> GetCompletedTrials() => _completedTrials;

    public void ReportBestTrial(TrialResult result)
    {
        var trialId = result.TrialSettings.TrialId;
        var timeToTrain = result.DurationInMilliseconds;
        var pipeline = _pipeline.ToString(result.TrialSettings.Parameter);
        var score = result.Metric;

        _richTextBox.AppendText($"Best trial {trialId} finished training in {timeToTrain}ms with pipeline {pipeline}, metric: {score}" + Environment.NewLine);
    }

    public void ReportCompletedTrial(TrialResult result)
    {
        var trialId = result.TrialSettings.TrialId;
        var timeToTrain = result.DurationInMilliseconds;
        var pipeline = _pipeline.ToString(result.TrialSettings.Parameter);
        var score = result.Metric;
        _richTextBox.AppendText($"Trial {trialId} finished training in {timeToTrain}ms with pipeline {pipeline}, metric: {score}" + Environment.NewLine);
    }

    public void ReportFailTrial(TrialSettings settings, Exception exception = null)
    {
        if (exception.Message.Contains("Operation was canceled."))
        {
            // logging to RichTextBox instead
            // Console.WriteLine($"{settings.TrialId} cancelled. Time budget exceeded.");
            _richTextBox.AppendText($"{settings.TrialId} cancelled. Time budget exceeded." + Environment.NewLine);
        }
        // logging to RichTextBox instead
        // Console.WriteLine($"{settings.TrialId} failed with exception {exception.Message}");
        _richTextBox.AppendText($"{settings.TrialId} failed with exception {exception.Message}" + Environment.NewLine);
    }

    public void ReportRunningTrial(TrialSettings setting)
    {
        return;
    }
}
