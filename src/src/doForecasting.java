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
            PrintWriter resultLog = new PrintWriter(new FileWriter("/home/cycle/workspace/wekaforecasting-new-features/results.txt", true));

            long startTime = System.currentTimeMillis();
            WekaForecaster forecaster = new WekaForecaster();
            myHashMap hashMap = new myHashMap();
            /*for (int i = 1; i < 1000; i+=48) {
                hashMap.fillUpHashMap(applyFilterClassifier.applyFilterClassifier(data, i, i+47), 4, hashMap, data.attribute(1).name());
            }
            myHashMap.sortHashMapByValues(hashMap);
            myHashMap.printHashMapFeatures(hashMap, 100);*/
            /*forecaster.getTSLagMaker().setTimeStampField(data.attribute(0).name()); // date time stamp
            forecaster.setFieldsToForecast(data.attribute(1).name());
            forecaster.setOverlayFields(data.attribute(2).name());
            forecaster.setBaseForecaster(classifier);
            forecaster.getTSLagMaker().setIncludePowersOfTime(true);
            forecaster.getTSLagMaker().setIncludeTimeLagProducts(false);
            //forecaster.getTSLagMaker().setFieldsToLagAsString(data.attribute(1).name() + ", " + data.attribute(2).name());
            forecaster.getTSLagMaker().setMinLag(1);
            forecaster.getTSLagMaker().setMaxLag(10);
            //forecaster.getTSLagMaker().setLagRange("1, 2, 4, 8, 12, 96, 672, 20, 576, 384, 480, 192");
            crossValidateTS(data, forecaster);*/
            TSLagMaker tsLagMaker = new TSLagMaker();
            tsLagMaker.setFieldsToLagAsString(data.attribute(1).name());
            tsLagMaker.setTimeStampField(data.attribute(0).name());
            tsLagMaker.setIncludePowersOfTime(true);
            tsLagMaker.setIncludeTimeLagProducts(false);
            tsLagMaker.setMinLag(1);
            tsLagMaker.setMaxLag(12);
            List<String> overlayFields = new ArrayList<String>();
            for (int i = 0; i < 4; i++)
                overlayFields.add(0, data.attribute(i+2).name());
            tsLagMaker.setOverlayFields(overlayFields);
            Instances laggedData = tsLagMaker.getTransformedData(data);
            //System.out.println(laggedData);
            BestFirst2 bestFirst2 = new BestFirst2();
            bestFirst2.search(laggedData);
            //map = fillUpHashMap(forecaster, featureNumber, map);
            //sortHashMapByValues(map);
            //printHashMapFeatures(map, featureNumber);

            resultLog.println(forecaster);
            calculateErrors(resultLog, true);

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
            for (int trainingPercentage = 80; trainingPercentage <= 80; trainingPercentage += 5) {
                long sTime = System.currentTimeMillis();
                trainData = getSplittedData(data, trainingPercentage, true);
                testData = getSplittedData(data, trainingPercentage, false);
                forecaster.buildForecaster(trainData);
                forecaster.primeForecaster(trainData);
                forecast = forecaster.forecast(stepNumber, testData);
                //System.out.println(forecaster.getTSLagMaker().getTransformedData(testData));
                addToValuesLists(forecast, testData, stepNumber);
                long eTime = System.currentTimeMillis();
                System.out.println(((double)(eTime-sTime))/1000);
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
                    " Act" +
                    ": " + df.format(errorSum / (i + 1)) + " RMSE:" + df.format(Math.sqrt(squaredErrorSum / (i + 1))) +
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