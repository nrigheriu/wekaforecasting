package src;

import com.sun.org.apache.xpath.internal.SourceTree;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.ReliefFAttributeEval;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.NumericPrediction;
import weka.classifiers.timeseries.WekaForecaster;
import weka.classifiers.functions.*;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.supervised.attribute.TSLagMaker;

import java.io.*;
import java.text.DecimalFormat;
import java.util.*;

/**
 * Created by cycle on 09.12.16.
 */
public class doForecasting {
    static List<Double> actualValuesList = new ArrayList<>();
    static List<Double> forecastedValuesList = new ArrayList<>();
    static HashMap<Integer, Float> map = new HashMap<Integer, Float>();


    public static void doForecasting(Instances data, Classifier classifier){
        try {
            PrintWriter resultLog = new PrintWriter(new FileWriter("/afs/tu-berlin.de/home/n/n_righeriu/irb-ubuntu/workspace/new_results.txt", true));

            long startTime = System.currentTimeMillis();
            WekaForecaster forecaster = new WekaForecaster();
            /*myHashMap hashMap = new myHashMap();
            for (int i = 1; i < 440;i+=48) {
                hashMap.fillUpHashMap(applyFilterClassifier.applyFilterClassifier(data, i, i+47), 5, hashMap, data.attribute(1).name());
            }
            myHashMap.sortHashMapByValues(hashMap);
            String chosenLags = myHashMap.printHashMapFeatures(hashMap, 50);*/


            MLPRegressor mlpRegressor = new MLPRegressor();
            forecaster.setBaseForecaster(classifier);
            forecaster.getTSLagMaker().setIncludePowersOfTime(true);
            forecaster.getTSLagMaker().setIncludeTimeLagProducts(false);
            forecaster.getTSLagMaker().setMinLag(1);
            forecaster.getTSLagMaker().setMaxLag(1344);

            forecaster.setFieldsToForecast(data.attribute(1).name());
            forecaster.getTSLagMaker().setTimeStampField(data.attribute(0).name());
            forecaster.getTSLagMaker().setLagRange("1-96");
            /*System.out.println("Chosen lags: " + chosenLags.substring(0, chosenLags.length()-2));
            forecaster.getTSLagMaker().setLagRange(chosenLags.substring(0, chosenLags.length()-2));*/
            crossValidateTS(data, forecaster);
            calculateErrors(resultLog, true);
            resultLog.println(forecaster);

            long stopTime = System.currentTimeMillis();
            double elapsedTime = ((double) stopTime - startTime)/1000;
            resultLog.println("Time taken: " + elapsedTime);
            resultLog.close();
        }catch (Exception e){
            e.printStackTrace();
        }
    }
    public static void crossValidateTS(Instances data, WekaForecaster forecaster){
        try {

            actualValuesList.clear();
            forecastedValuesList.clear();
            int stepNumber = 24;
            Instances testData = null, trainData = null;
            List<List<NumericPrediction>> forecast = null;
            for (int trainingPercentage = 70; trainingPercentage <= 80; trainingPercentage += 5) {
                trainData = getSplittedData(data, trainingPercentage, true);
                testData = getSplittedData(data, trainingPercentage, false);
                //forecaster.setOverlayFields(data.attribute(2).name());

                forecaster.buildForecaster(trainData);
                forecaster.primeForecaster(trainData);
                forecast = forecaster.forecast(stepNumber);

                addToValuesLists(forecast, testData, stepNumber);
            }
            buildErrorGraph.buildErrorGraph(testData, forecaster, forecast, stepNumber);
        } catch (Exception e){
            e.printStackTrace();
        }
    }
    public static void addToValuesLists(List<List<NumericPrediction>> forecast, Instances testData, int stepNumber){
        for (int i = 0; i < stepNumber; i++) {
            actualValuesList.add(testData.get(i).value(1));
            forecastedValuesList.add(forecast.get(i).get(0).predicted());
        }
    }
    public static Instances getSplittedData(Instances data, Integer trainPercent, boolean getTrainData){
        int trainSize = (int) Math.round(data.numInstances() * trainPercent/100);
        int testSize = data.numInstances()-trainSize;
        if (getTrainData){
            return new Instances(data, 0, trainSize);
        }else {
            return new Instances(data, trainSize, testSize);
        }
    }
    public static float calculateErrors (PrintWriter resultLog, boolean writeToLog){
        double errorSum = 0;
        double piErrorSum = 0;
        double squaredErrorSum = 0;
        DecimalFormat df = new DecimalFormat("#.###");
        List<String> errorList = new ArrayList<>();
        float getLastError = 0;
        for (int i = 0; i < actualValuesList.size(); i++) {
            double actualValue = actualValuesList.get(i);
            double error = Math.abs(forecastedValuesList.get(i) - actualValue);
            double piError = 100 * error / actualValue;
            piErrorSum += piError;
            errorSum += error;
            squaredErrorSum += error*error;
            String errorOutput = "Step: " + i + " Prediction:" + df.format(forecastedValuesList.get(i)) +
                    " Actual: " + actualValue +
                    " MAE: " + df.format(errorSum / (i + 1)) + " RMSE:" + df.format(Math.sqrt(squaredErrorSum / (i + 1))) +
                    " MAPE:" + df.format(piErrorSum / (i + 1));
            if(writeToLog)
                resultLog.println(errorOutput);
            getLastError = (float) Math.sqrt(squaredErrorSum/ (i + 1));
        }
        return getLastError;
    }

    public static Float getAvg(Float[] array){
        Float avg = (float) 0;
        for (int i = 0; i< array.length; i++)
            avg += array[i];
        return avg/array.length;
    }
}


/*TSEvaluation evaluation = new TSEvaluation(testData, stepNumber);
            evaluation.setEvaluateOnTestData(true);
            evaluation.setEvaluateOnTrainingData(false);
            evaluation.setForecastFuture(true);
            evaluation.setHorizon(stepNumber);
            evaluation.setEvaluationModules("MAE, MAPE, RMSE");
            evaluation.evaluateForecaster(forecaster);
            System.out.println(evaluation.toSummaryString());
            System.out.println(evaluation.getPredictionsForTestData(stepNumber));*/
//resultLog.println(forecaster.getTSLagMaker().createTimeLagCrossProducts(data) + "\n");
//resultLog.println(forecaster.getTSLagMaker().getTransformedData(data) + "\n");

           /* //map = sortHashMapByValues(map);
            Set<Integer> mapKeys = map.keySet();
            String combinedFeatures = "";
            int j = 1;
            for(Integer key:mapKeys){
                if(j > featureNumber)
                    break;
                combinedFeatures += String.valueOf(key) + ", ";
                j++;
            }
            System.out.println(combinedFeatures);*/

//forecaster.getTSLagMaker().setAddDayOfWeek(true);
//forecaster.getTSLagMaker().setAddMonthOfYear(true);

/*     forecaster.getTSLagMaker().setTimeStampField(data.attribute(0).name()); // date time stamp
            forecaster.setFieldsToForecast(data.attribute(1).name());
            //forecaster.setOverlayFields(data.attribute(2).name());
            forecaster.setBaseForecaster(classifier);
            forecaster.getTSLagMaker().setIncludePowersOfTime(true);
            forecaster.getTSLagMaker().setIncludeTimeLagProducts(false);

            //crossValidateTS(data, forecaster);

            int featureNumber = 50;
            myHashMap myMap = new myHashMap();
            rankerWrapper rankerWrapperObj = new rankerWrapper();
            Float[] percentFeaturesToGetFromInterval = rankerWrapperObj.getPercentagesForIntervals(forecaster, data, resultLog);
            ArrayList<ArrayList<Integer>> featureListForIntevals = rankerWrapperObj.featureListForIntevals;
            ArrayList<Integer> selectedFeatures = rankerWrapperObj.populateSelectedFeatures(featureListForIntevals, percentFeaturesToGetFromInterval, 10);

            forecaster.setBaseForecaster(new MLPRegressor());                 //running classifier on attributes ranked by rankedWrapper
            forecaster.getTSLagMaker().setMinLag(1);
            forecaster.getTSLagMaker().setMaxLag(780);
            resultLog.println(selectedFeatures.toString());
            forecaster.getTSLagMaker().setLagRange(selectedFeatures.toString().substring(1, selectedFeatures.toString().length()-1)); */