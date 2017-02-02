package src;

import weka.classifiers.evaluation.NumericPrediction;
import weka.classifiers.timeseries.WekaForecaster;
import weka.core.Instances;
import weka.classifiers.Classifier;
import weka.filters.supervised.attribute.TSLagMaker;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by cycle on 13.01.2017.
 */
public class TSCV {
        private List<Double> actualValuesList = new ArrayList<>();
        private List<Double> forecastedValuesList = new ArrayList<>();
        public TSCV(){
            resetOptions();
        }
    public void crossValidateTS(Instances data, Classifier classifier, TSLagMaker tsLagMaker){
        try {
            PrintWriter resultLog = new PrintWriter(new FileWriter("/home/cycle/workspace/wekaforecasting-new-features/results.txt", true));
            actualValuesList.clear();
            forecastedValuesList.clear();
            WekaForecaster forecaster = new WekaForecaster();
            forecaster.setTSLagMaker(tsLagMaker);
            forecaster.setFieldsToForecast(tsLagMaker.getFieldsToLagAsString());
            forecaster.setBaseForecaster(classifier);
            int stepNumber = 24;
            Instances testData = null, trainData = null;
            List<List<NumericPrediction>> forecast = null;
            //System.out.println("Lag range: " + tsLagMaker.getLagRange());
            for (int trainingPercentage = 80; trainingPercentage <= 80; trainingPercentage += 5) {
                long sTime = System.currentTimeMillis();
                trainData = getSplittedData(data, trainingPercentage, true);
                testData = getSplittedData(data, trainingPercentage, false);
                forecaster.buildForecaster(trainData);
                forecaster.primeForecaster(trainData);
               if(!forecaster.getTSLagMaker().getOverlayFields().isEmpty())                        //checking if any overlay fields are set
                    forecast = forecaster.forecast(stepNumber, testData);
                else
                    forecast = forecaster.forecast(stepNumber);
                addToValuesLists(forecast, testData, stepNumber);
                long eTime = System.currentTimeMillis();
                //System.out.println(("Time taken to evaluate: " + ((double)(eTime-sTime))/1000));
                resultLog.close();
            }
            buildErrorGraph.buildErrorGraph(testData, forecaster, forecast, stepNumber);
        } catch (Exception e){
            e.printStackTrace();
        }
    }
    public void addToValuesLists(List<List<NumericPrediction>> forecast, Instances testData, int stepNumber){
        for (int i = 0; i < stepNumber; i++) {
            actualValuesList.add(testData.get(i).value(1));
            forecastedValuesList.add(forecast.get(i).get(0).predicted());
        }
    }
    public Instances getSplittedData(Instances data, Integer trainPercent, boolean getTrainData){
        int trainSize = (int) Math.round(data.numInstances() * trainPercent/100);
        int testSize = data.numInstances()-trainSize;
        if (getTrainData){
            return new Instances(data, 0, trainSize);
        }else {
            return new Instances(data, trainSize, testSize);
        }
    }
    public double calculateErrors (boolean printOutput, String evaluationMeasure){
        double errorSum = 0;
        double piErrorSum = 0;
        double squaredErrorSum = 0;
        DecimalFormat df = new DecimalFormat("#.###");
        List<String> errorList = new ArrayList<>();
        double getLastError = 0;
        int i;
        for (i = 0; i < actualValuesList.size(); i++) {
            double actualValue = actualValuesList.get(i);
            double error = Math.abs(forecastedValuesList.get(i) - actualValue);
            double piError = 100 * error / actualValue;
            piErrorSum += piError;
            errorSum += error;
            squaredErrorSum += error*error;
            String errorOutput = "Step: " + i + " Prediction:" + df.format(forecastedValuesList.get(i)) +
                    " Act: " + actualValue +
                    " MAE: " + df.format(errorSum / (i + 1)) + " RMSE:" + df.format(Math.sqrt(squaredErrorSum / (i + 1))) +
                    " MAPE:" + df.format(piErrorSum / (i + 1));
            if(printOutput)
                System.out.println(errorOutput);
        }
        if(evaluationMeasure == "RMSE")
            getLastError = Math.sqrt(squaredErrorSum/ (i + 1));
        else if (evaluationMeasure == "MAPE")
            getLastError = piErrorSum/(i+1);
        return getLastError;
    }
    protected void resetOptions(){
        actualValuesList.clear();
        forecastedValuesList.clear();
    }
}
