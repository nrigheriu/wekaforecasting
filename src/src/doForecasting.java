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
    public int threadNumber;

    public doForecasting() {
        resetOptions();
    }

    public void doForecast(Instances data) {
        try {
            //PrintWriter resultLog = new PrintWriter(new FileWriter("results.txt", true));
            this.threadNumber = 6;
            long startTime = System.currentTimeMillis(), stopTime;
            List<String> overlayFields = new ArrayList<String>();
            MyHashMap hashMap = new MyHashMap();
            TSLagMaker tsLagMaker = new TSLagMaker();
            int lagLimit = 1392, lagInterval = 36, featureLimitFromInterval = 3, reliefFeatureCutOff = 150;

           /* rankWithRelief(hashMap, data, lagInterval, lagLimit, featureLimitFromInterval);
            hashMap.sortHashMapByValues();
            String chosenLags = hashMap.printHashMapFeatures(reliefFeatureCutOff);*/
            String chosenLags = "768, 1340, 1, 1338, 1339, 766, 767, 1345, 1344, 1341, 1237, 672, 673, 1236, 962, 964, 961, 671, 1233, 844, 1041, 1040, 845, 864, 1045, 572, 1148, 1149, 1145, 769, 1144, 484, 571, 772, 570, 481, 479, 771, 1232, 1143, 1139, 1230, 1231, 573, 965, 576, 96, 968, 577, 95, 2, 948, 94, 969, 946, 189, 191, 190, 1161, 947, 1164, 465, 480, 6, 1163, 376, 287, 288, 286, 380, 377, 381, 1036, 877, 681, 1034, 1035, 180, 375, 880, 881, 840, 374, 1073, 1076, 684, 838, 1077, 839, 685, 1272, 1269, 179, 1273, 178, 730, 731, 732, 1381, 1380, 268, 1392, 109, 267, 644, 643, 266, 72, 642, 111, 110, 71, 70, 305, 308, 448, 447, 501, 309, 446, 504, 503, 252, 251, 250";
            System.out.println("Relief configuration, lagLimit:" + lagLimit
                    + " lagInterval:" + lagInterval + " featureLimitFromInterval:" + featureLimitFromInterval
            + " reliefFeatureCutOff:" + reliefFeatureCutOff);
            stopTime = System.currentTimeMillis();
            System.out.println("Time taken to rank lags w/Rrelief:" + ((double)(stopTime-startTime))/1000);
            System.out.println("Lags chosen by relief:" + chosenLags);
            tsLagMaker.setLagRange(chosenLags);
            /*System.out.println("Manual feature set with lags:1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 96, 192, 288, 672");
            tsLagMaker.setLagRange("1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 96, 192, 288, 672");*/
            tsLagMaker.setFieldsToLagAsString(data.attribute(1).name());
            tsLagMaker.setTimeStampField(data.attribute(0).name());
            tsLagMaker.setIncludePowersOfTime(true);
            tsLagMaker.setIncludeTimeLagProducts(false);
            tsLagMaker.setMinLag(1);
            tsLagMaker.setMaxLag(lagLimit);
            //tsLagMaker.setLagRange("937, 984, 983, 938, 745, 792, 791, 790, 841, 888, 887, 886, 1333, 1334, 1380, 1335, 588, 587, 1091, 1045, 1089, 586, 1092, 1079, 541, 1284, 1237, 1141, 1077, 1080, 1283, 1282, 1078, 1032, 1142, 683, 1332, 684, 1188, 1031, 985, 1143, 1030, 681, 682, 1331, 1330, 1329, 1140, 1236, 1139, 1235, 1138, 1137, 1234, 1233, 936, 935, 934, 1381, 933, 1392, 1382, 1383, 685, 744, 743, 697, 686, 493, 742, 732, 731, 396, 494");
            //tsLagMaker.setLagRange("697, 792, 888, 1030, 1032, 1080, 1333, 588");
            //tsLagMaker.setLagRange("768, 1, 769, 2, 3, 4, 1291, 1292, 527, 528, 1296, 1049, 282, 1051, 286, 289, 290, 1058, 814, 815, 816, 817, 573, 1341, 574, 1342, 575, 1343, 576, 1344, 577, 578, 579, 1102, 335, 1103, 336, 1104, 93, 94, 862, 95, 863, 96, 864, 97, 865, 98, 99, 101, 1389, 1390, 1391, 624, 1392, 381, 1149, 382, 1150, 383, 1151, 384, 1152, 385, 910, 911, 912, 914, 668, 669, 671");
            for (int i = 0; i < data.numAttributes() - 2; i++)                                        //first 2 attributes are time and field to lag
                overlayFields.add(i, data.attribute(i + 2).name());
            tsLagMaker.setOverlayFields(overlayFields);
            Instances laggedData = tsLagMaker.getTransformedData(data);
            src.BestFirst bestFirst = new src.BestFirst();
            //bestFirst.setOptions(weka.core.Utils.splitOptions("-D 2"));
            SimmulatedAnnealing simmulatedAnnealing = new SimmulatedAnnealing();
            RandomSearch randomSearch = new RandomSearch();
            randomSearch.setThreadNumber(threadNumber);
            randomSearch.search(laggedData, tsLagMaker, overlayFields);
            //simmulatedAnnealing.search(laggedData, tsLagMaker, overlayFields);
            //bestFirst.search(laggedData, tsLagMaker, overlayFields);
            stopTime = System.currentTimeMillis();
            System.out.println("Time taken for all:" + ((double) stopTime - startTime) / 1000);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void rankWithRelief(MyHashMap hashMap, Instances data, int lagInterval, int lagLimit, int featureNumber) {
        int startLag = 1, endLag = startLag + lagInterval - 1;
        ArrayList<Thread> threadList = new ArrayList<Thread>();
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