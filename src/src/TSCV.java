package src;

import com.sun.org.apache.xpath.internal.SourceTree;
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
    public void crossValidateTS(Instances data, Classifier classifier, TSLagMaker tsLagMaker, boolean testBestModel){
        try {
            actualValuesList.clear();
            forecastedValuesList.clear();
            WekaForecaster forecaster = new WekaForecaster();
            forecaster.setTSLagMaker(tsLagMaker);
            forecaster.setFieldsToForecast(tsLagMaker.getFieldsToLagAsString());
            forecaster.setBaseForecaster(classifier);
            int stepNumber = 24, trainingPercentage;                          //stepNumber is how many steps in future it should be forecasted. 15 min * 24 =  6 hours
            int numberOfUnitsToForecast, numUnitsForecasted, daysToForecast;   //this specifies how many times the stepNumber above should be evaluated with the same forecaster built.
            ArrayList<Integer> trainingPercentages = new ArrayList<Integer>();
            if(testBestModel){
                for (int i = 80; i <= 97; i+=3)      //3% of 1year and 3 months is 2 weeks
                    trainingPercentages.add(i);
                trainingPercentage = 80;
                daysToForecast = 14;
            }else {
                trainingPercentages.add(70);
                trainingPercentage = 70;
                daysToForecast = 7;
            }
            numberOfUnitsToForecast = (daysToForecast * 24)/6;
            for (int i = 0; i < trainingPercentages.size(); i++){
                trainingPercentage = trainingPercentages.get(i);
                List<List<NumericPrediction>> forecast = null;
                Instances testData = null, trainData = null;
                int startTestData = 0, endTestData = 0;
                long sTime = System.currentTimeMillis();
                numUnitsForecasted = 1;
                trainData = getSplittedData(data, trainingPercentage, true);
                testData = getSplittedData(data, trainingPercentage, false);
                forecaster.buildForecaster(trainData);
                forecaster.primeForecaster(trainData);
                //numberOfUnitsToForecast = (int) Math.floor(testData.numInstances() / stepNumber);                                               //forecast until the end of the data set; can take a whole while longer
                while (numUnitsForecasted <= numberOfUnitsToForecast) {
                    startTestData = (numUnitsForecasted - 1) * (stepNumber);
                    endTestData = (int) testData.numInstances() - startTestData;
                    if (!forecaster.getTSLagMaker().getOverlayFields().isEmpty())                        //checking if any overlay fields are set
                        forecast = forecaster.forecast(stepNumber, new Instances(testData, startTestData, endTestData));
                    else
                        forecast = forecaster.forecast(stepNumber);
                    forecaster.primeForecaster(trainData);
                    addToValuesLists(forecast, new Instances(testData, startTestData, endTestData), stepNumber);
                    if (numUnitsForecasted < numberOfUnitsToForecast - 1)                                     //check if this isn't the last iteration and where are priming for nothing
                        for (int j = 0; j < stepNumber * numUnitsForecasted; j++)
                            forecaster.primeForecasterIncremental(testData.get(j));
                    numUnitsForecasted++;
                }
                if(i == trainingPercentages.size() - 1 && testBestModel)
                    buildErrorGraph.buildErrorGraph(new Instances(testData, startTestData, endTestData), forecaster, forecast, stepNumber);
                long eTime = System.currentTimeMillis();
                System.out.println(("Time taken to evaluate again:" + ((double) (eTime - sTime)) / 1000));
            }
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
        if (getTrainData)
            return new Instances(data, 0, trainSize);
        else
            return new Instances(data, trainSize, testSize);
    }
    public double calculateErrors (boolean printOutput, String evaluationMeasure){
        //mode = 0 for wrapper, 1 for returning bias and 2 for testing best model
        double errorSum = 0;
        double piErrorSum = 0;
        double squaredErrorSum = 0;
        DecimalFormat df = new DecimalFormat("#.####");
        double getLastError = 0;
        int i;
        for (i = 0; i < actualValuesList.size(); i++) {
            double actualValue = actualValuesList.get(i);
            double forecastedValue;
            forecastedValue = forecastedValuesList.get(i);
            double error = Math.abs(forecastedValue - actualValue);
            double piError = 100 * error / actualValue;
            piErrorSum += piError;
            errorSum += error;
            squaredErrorSum += error*error;
            String errorOutput = "Step: " + i + " Prediction: " + df.format(forecastedValue) +
                    " Act:" + df.format(actualValue) +
                    " MAE:" + df.format(errorSum / (i + 1)) + " RMSE:" + df.format(Math.sqrt(squaredErrorSum / (i + 1))) +
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



/*
        Double sum = 0.;
        if(mode == 2) {
                System.out.println("No mean adjustment!");
                for (i = 0; i < actualValuesList.size(); i++) {
        double actualValue = actualValuesList.get(i);
        double forecastedValue;
        forecastedValue = forecastedValuesList.get(i);
        double error = Math.abs(forecastedValue - actualValue);
        double piError = 100 * error / actualValue;
        piErrorSum += piError;
        errorSum += error;
        squaredErrorSum += error * error;
        String errorOutput = "Step: " + i + " Prediction: " + df.format(forecastedValue) +
        " Act:" + df.format(actualValue) +
        " MAE:" + df.format(errorSum / (i + 1)) + " RMSE:" + df.format(Math.sqrt(squaredErrorSum / (i + 1))) +
        " MAPE:" + df.format(piErrorSum / (i + 1));
        if (printOutput)
        System.out.println(errorOutput);
        }
        System.out.println("With mean adjustment:");
        errorSum = 0;
        piErrorSum = 0;
        squaredErrorSum = 0;
        getLastError = 0;
        }
         if(mode == 2){
            sum = 0.;
            for (int j = 0; j < actualValuesList.size(); j++)
                sum += actualValuesList.get(j) - forecastedValuesList.get(j);
            System.out.println("Sum @ end:" + sum);
        }
        */
