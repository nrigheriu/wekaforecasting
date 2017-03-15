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
        private String searchMethod = "";
        public TSCV(){
            resetOptions();
        }
    public void crossValidateTS(Instances data, Classifier classifier, TSLagMaker tsLagMaker){
        try {
            actualValuesList.clear();
            forecastedValuesList.clear();
            WekaForecaster forecaster = new WekaForecaster();
            forecaster.setTSLagMaker(tsLagMaker);
            forecaster.setFieldsToForecast(tsLagMaker.getFieldsToLagAsString());
            forecaster.setBaseForecaster(classifier);
            int stepNumber = 24;                           //stepNumber is how many steps in future it should be forecasted. 15 min * 24 =  6 hours
            int numberOfUnitsToForecast, numUnitsForecasted;               //this specifies how many times the stepNumber above should be evaluated with the same forecaster built.
            int startTestData = 0, endTestData = 0;
            Instances testData = null, trainData = null;
            List<List<NumericPrediction>> forecast = null;
            long sTime = System.currentTimeMillis();
            int[] trainingPercentages = {65, 70, 75};
            for (int trainingPercentage:trainingPercentages) {
                numUnitsForecasted = 1;
                numberOfUnitsToForecast = 56;
                trainData = getSplittedData(data, trainingPercentage, true);
                testData = getSplittedData(data, trainingPercentage, false);
                forecaster.buildForecaster(trainData);
                forecaster.primeForecaster(trainData);
                //numberOfUnitsToForecast = (int) Math.floor(testData.numInstances() / stepNumber);                                               //forecast until the end of the data set; can take a whole while longer
                while (numUnitsForecasted <= numberOfUnitsToForecast){
                    startTestData = (numUnitsForecasted-1)*(stepNumber);
                    endTestData = (int)testData.numInstances()-startTestData;
                    if(!forecaster.getTSLagMaker().getOverlayFields().isEmpty())                        //checking if any overlay fields are set
                        forecast = forecaster.forecast(stepNumber, new Instances(testData, startTestData, endTestData));
                    else
                        forecast = forecaster.forecast(stepNumber);
                    forecaster.primeForecaster(trainData);
                    addToValuesLists(forecast, new Instances(testData, startTestData, endTestData), stepNumber);
                    if(numUnitsForecasted < numberOfUnitsToForecast -1)                                     //check if this isn't the last iteration and where are priming for nothing
                        for (int i = 0; i < stepNumber*numUnitsForecasted; i++)
                            forecaster.primeForecasterIncremental(testData.get(i));
                    numUnitsForecasted++;
                }
                long eTime = System.currentTimeMillis();
                System.out.println(("Time taken to evaluate again:" + ((double)(eTime-sTime))/1000));
            }
            buildErrorGraph.buildErrorGraph(new Instances(testData, startTestData, endTestData), forecaster, forecast, stepNumber);
        } catch (Exception e){
            e.printStackTrace();
        }
    }
    public void testBestModel(Instances data, Classifier classifier, TSLagMaker tsLagMaker){
        try {
            resetOptions();
            WekaForecaster forecaster = new WekaForecaster();
            forecaster.setTSLagMaker(tsLagMaker);
            forecaster.setFieldsToForecast(tsLagMaker.getFieldsToLagAsString());
            forecaster.setBaseForecaster(classifier);
            int numberOfUnitsToForecast, numUnitsForecasted;               //this specifies how many times the stepNumber above should be evaluated with the same forecaster built.
            int stepNumber = 24;
            int startTestData = 0, endTestData = 0;
            Instances testData = null, trainData = null;
            List<List<NumericPrediction>> forecast = null;
            long sTime = System.currentTimeMillis();
            int[] trainingPercentages = {80};
            for (int trainingPercentage:trainingPercentages) {
                numUnitsForecasted = 1;
                numberOfUnitsToForecast = 56;
                trainData = getSplittedData(data, trainingPercentage, true);
                testData = getSplittedData(data, trainingPercentage, false);
                forecaster.buildForecaster(trainData);
                forecaster.primeForecaster(trainData);
                while (numUnitsForecasted <= numberOfUnitsToForecast){
                    startTestData = (numUnitsForecasted-1)*(stepNumber);
                    endTestData = (int)testData.numInstances()-startTestData;
                    if(!forecaster.getTSLagMaker().getOverlayFields().isEmpty())                        //checking if any overlay fields are set
                        forecast = forecaster.forecast(stepNumber, new Instances(testData, startTestData, endTestData));
                    else
                        forecast = forecaster.forecast(stepNumber);
                    forecaster.primeForecaster(trainData);
                    addToValuesLists(forecast, new Instances(testData, startTestData, endTestData), stepNumber);
                    if(numUnitsForecasted < numberOfUnitsToForecast -1)                                     //check if this isn't the last iteration and where are priming for nothing
                        for (int i = 0; i < stepNumber*numUnitsForecasted; i++)
                            forecaster.primeForecasterIncremental(testData.get(i));
                    numUnitsForecasted++;
                }
                long eTime = System.currentTimeMillis();

                System.out.println(("Time taken to evaluate final model:" + ((double)(eTime-sTime))/1000));
            }
        }catch (Exception e){
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
        if (getTrainData)
            return new Instances(data, 0, trainSize);
        else
            return new Instances(data, trainSize, testSize);
    }
    public double calculateErrors (boolean printOutput, String evaluationMeasure){
        double errorSum = 0;
        double piErrorSum = 0;
        double squaredErrorSum = 0;
        DecimalFormat df = new DecimalFormat("#.####");
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
    public String getSearchMethod() {
        return searchMethod;
    }
    public void setSearchMethod(String searchMethod) {
        this.searchMethod = searchMethod;
    }
    protected void resetOptions(){
        actualValuesList.clear();
        forecastedValuesList.clear();
    }
}
