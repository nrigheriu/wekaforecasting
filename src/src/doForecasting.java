package src;

import jdk.nashorn.internal.runtime.regexp.joni.exception.ValueException;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.NumericPrediction;
import weka.core.Instances;
import weka.filters.supervised.attribute.TSLagMaker;
import weka.classifiers.timeseries.WekaForecaster;

import java.io.*;
import java.text.DecimalFormat;
import java.util.*;

/**
 * Created by cycle on 09.12.16.
 */
public class doForecasting {
    List<Double> actualValuesList = new ArrayList<>();
    List<Double> forecastedValuesList = new ArrayList<>();
    HashMap<Integer, Float> map = new HashMap<Integer, Float>();
    private Thread t;
    private String threadName;

    public doForecasting() {
        resetOptions();
    }

    public void doForecast(Instances data) {
        try {               //TODO:change folder of results file corresponding to the used method
            PrintWriter resultLog = new PrintWriter(new FileWriter("results.txt", true));
            long startTime = System.currentTimeMillis();
            List<String> overlayFields = new ArrayList<String>();
            MyHashMap hashMap = new MyHashMap();
            int lagLimit = 192, lagInterval = 48, featureLimitFromInterval = 48, reliefFeatureCutOff = 75;


            rankWithRelief(hashMap, data, lagInterval, lagLimit, featureLimitFromInterval);
            hashMap.sortHashMapByValues();
            String chosenLags = hashMap.printHashMapFeatures(reliefFeatureCutOff);
            resultLog.println("Relief configuration, lagLimit:" + lagLimit
                    + " lagInterval:" + lagInterval + " featureLimitFromInterval:" + featureLimitFromInterval
            + "reliefFeatureCutOff:" + reliefFeatureCutOff);
            resultLog.println("Lags chosen by relief:" + chosenLags);

            TSLagMaker tsLagMaker = new TSLagMaker();
            tsLagMaker.setFieldsToLagAsString(data.attribute(1).name());
            tsLagMaker.setTimeStampField(data.attribute(0).name());
            tsLagMaker.setIncludePowersOfTime(true);
            tsLagMaker.setIncludeTimeLagProducts(false);
            tsLagMaker.setMinLag(1);
            tsLagMaker.setMaxLag(lagLimit);
            tsLagMaker.setLagRange(chosenLags);
            //tsLagMaker.setLagRange("937, 984, 983, 938, 745, 792, 791, 790, 841, 888, 887, 886, 1333, 1334, 1380, 1335, 588, 587, 1091, 1045, 1089, 586, 1092, 1079, 541, 1284, 1237, 1141, 1077, 1080, 1283, 1282, 1078, 1032, 1142, 683, 1332, 684, 1188, 1031, 985, 1143, 1030, 681, 682, 1331, 1330, 1329, 1140, 1236, 1139, 1235, 1138, 1137, 1234, 1233, 936, 935, 934, 1381, 933, 1392, 1382, 1383, 685, 744, 743, 697, 686, 493, 742, 732, 731, 396, 494");
            //tsLagMaker.setLagRange("697, 792, 888, 1030, 1032, 1080, 1333, 588");
            for (int i = 0; i < data.numAttributes() - 2; i++)                                        //first 2 attributes are time and field to lag
                overlayFields.add(i, data.attribute(i + 2).name());
            tsLagMaker.setOverlayFields(overlayFields);
            Instances laggedData = tsLagMaker.getTransformedData(data);
            src.BestFirst bestFirst = new src.BestFirst();
            //bestFirst.setOptions(weka.core.Utils.splitOptions("-D 2"));
            SimmulatedAnnealing simmulatedAnnealing = new SimmulatedAnnealing();
            RandomSearch randomSearch = new RandomSearch();

            randomSearch.search(laggedData, tsLagMaker, overlayFields);
            //simmulatedAnnealing.search(laggedData, tsLagMaker, overlayFields);
            // tsLagMaker.setLagRange("768, 1, 769, 2, 3, 4, 1291, 1292, 527, 528, 1296, 1049, 282, 1051, 286, 289, 290, 1058, 814, 815, 816, 817, 573, 1341, 574, 1342, 575, 1343, 576, 1344, 577, 578, 579, 1102, 335, 1103, 336, 1104, 93, 94, 862, 95, 863, 96, 864, 97, 865, 98, 99, 101, 1389, 1390, 1391, 624, 1392, 381, 1149, 382, 1150, 383, 1151, 384, 1152, 385, 910, 911, 912, 914, 668, 669, 671");
            //simmulatedAnnealing.search(laggedData, tsLagMaker, overlayFields);
            //bestFirst.search(laggedData, tsLagMaker, overlayFields);
            long stopTime = System.currentTimeMillis();
            double elapsedTime = ((double) stopTime - startTime) / 1000;
            System.out.println("Time taken: " + elapsedTime);
            resultLog.println("Time taken: " + elapsedTime);
            resultLog.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void rankWithRelief(MyHashMap hashMap, Instances data, int lagInterval, int lagLimit, int featureNumber) {
        int startLag = 1, endLag = startLag + lagInterval - 1;
        ArrayList<Thread> threadList = new ArrayList<Thread>();
        int threadNumber = 4;
        if (lagLimit % threadNumber != 0)
            throw new ValueException("Lag limit has to be divisible with " + threadNumber + " because the lag intervals will be split to that number of threads!");
        int threadLagInterval = lagLimit / threadNumber;                          //each thread has a Interval of the whole lag range which will be assigned to it; this Interval will be calculated by the thread split again in the per parameter given Interval size
        for (int i = 0; i < threadNumber; i++) {
            startLag = (threadLagInterval * i) + 1;
            endLag = threadLagInterval * (i + 1);
            System.out.println("Start lag:" + startLag + " endLag:" + endLag);
            threadList.add(i, new MyThread("Thread" + (i + 1), hashMap, data, startLag, endLag, featureNumber, lagLimit, lagInterval));
        }
        for (int i = 0; i < threadList.size(); i++)
            threadList.get(i).start();
        try {
            for (int i = 0; i < threadList.size(); i++)                     //waiting for all threads to finish
                threadList.get(i).join();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void crossValidateTS(Instances data, WekaForecaster forecaster) {
        try {
            this.actualValuesList.clear();
            forecastedValuesList.clear();
            int stepNumber = 24;
            Instances testData = null, trainData = null;
            List<List<NumericPrediction>> forecast = null;
            for (int trainingPercentage = 70; trainingPercentage <= 80; trainingPercentage += 5) {
                long sTime = System.currentTimeMillis();
                trainData = getSplittedData(data, trainingPercentage, true);
                testData = getSplittedData(data, trainingPercentage, false);
                forecaster.buildForecaster(trainData);
                forecaster.primeForecaster(trainData);
                if (!forecaster.getTSLagMaker().getOverlayFields().isEmpty())                        //checking if any overlay fields are set
                    forecast = forecaster.forecast(stepNumber, testData);
                else
                    forecast = forecaster.forecast(stepNumber);
                //System.out.println(forecaster.getTSLagMaker().getTransformedData(testData));
                addToValuesLists(forecast, testData, stepNumber);
                long eTime = System.currentTimeMillis();
                System.out.println(((double) (eTime - sTime)) / 1000);
            }
            buildErrorGraph.buildErrorGraph(testData, forecaster, forecast, stepNumber);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void addToValuesLists(List<List<NumericPrediction>> forecast, Instances testData, int stepNumber) {
        for (int i = 0; i < stepNumber; i++) {
            actualValuesList.add(testData.get(i).value(1));
            forecastedValuesList.add(forecast.get(i).get(0).predicted());
        }
    }

    public Instances getSplittedData(Instances data, Integer trainPercent, boolean getTrainData) {
        int trainSize = (int) Math.round(data.numInstances() * trainPercent / 100);
        int testSize = data.numInstances() - trainSize;
        if (getTrainData) {
            return new Instances(data, 0, trainSize);
        } else {
            return new Instances(data, trainSize, testSize);
        }
    }

    public double calculateErrors(boolean printOutput, String evaluationMeasure, PrintWriter resultLog) {
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
            squaredErrorSum += error * error;
            String errorOutput = "Step: " + i + " Prediction:" + df.format(forecastedValuesList.get(i)) +
                    " Act: " + actualValue +
                    " MAE: " + df.format(errorSum / (i + 1)) + " RMSE:" + df.format(Math.sqrt(squaredErrorSum / (i + 1))) +
                    " MAPE:" + df.format(piErrorSum / (i + 1));
            if (printOutput)
                resultLog.println(errorOutput);
        }
        if (evaluationMeasure == "RMSE")
            getLastError = Math.sqrt(squaredErrorSum / (i + 1));
        else if (evaluationMeasure == "MAPE")
            getLastError = piErrorSum / (i + 1);
        return getLastError;
    }

    public Float getAvg(Float[] array) {
        Float avg = (float) 0;
        for (int i = 0; i < array.length; i++)
            avg += array[i];
        return avg / array.length;
    }

    public void resetOptions() {
        actualValuesList = null;
        forecastedValuesList = null;
        map = null;
    }
}

    /* WekaForecaster forecaster = new WekaForecaster();
    for (int i = 0; i < data.numAttributes()-2; i++)                                        //first 2 attributes are time and field to lag
        overlayFields.add(i, data.attribute(i+2).name());
    forecaster.getTSLagMaker().setOverlayFields(overlayFields);
    forecaster.getTSLagMaker().setTimeStampField(data.attribute(0).name()); // date time stamp
    forecaster.setFieldsToForecast(data.attribute(1).name());
    forecaster.setBaseForecaster(classifier);
    forecaster.getTSLagMaker().setIncludePowersOfTime(true);
    forecaster.getTSLagMaker().setIncludeTimeLagProducts(false);
    forecaster.getTSLagMaker().setFieldsToLagAsString(data.attribute(1).name() + ", " + data.attribute(2).name());
    //crossValidateTS(data, forecaster);
    calculateErrors(true,  "MAPE", resultLog);*/

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
            MyHashMap myMap = new MyHashMap();
            rankerWrapper rankerWrapperObj = new rankerWrapper();
            Float[] percentFeaturesToGetFromInterval = rankerWrapperObj.getPercentagesForIntervals(forecaster, data, resultLog);
            ArrayList<ArrayList<Integer>> featureListForIntevals = rankerWrapperObj.featureListForIntevals;
            ArrayList<Integer> selectedFeatures = rankerWrapperObj.populateSelectedFeatures(featureListForIntevals, percentFeaturesToGetFromInterval, 10);

            forecaster.setBaseForecaster(new MLPRegressor());                 //running classifier on attributes ranked by rankedWrapper
            forecaster.getTSLagMaker().setStartLag(1);
            forecaster.getTSLagMaker().setEndLag(780);
            resultLog.println(selectedFeatures.toString());
            forecaster.getTSLagMaker().setLagRange(selectedFeatures.toString().substring(1, selectedFeatures.toString().length()-1)); */