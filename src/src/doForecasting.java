package src;

import weka.attributeSelection.*;
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
    public doForecasting(){
      resetOptions();
    }
    public void doForecasting(Instances data, Classifier classifier){
        try {
            PrintWriter resultLog = new PrintWriter(new FileWriter("/home/cycle/workspace/wekaforecasting-new-features/results.txt", true));
            long startTime = System.currentTimeMillis();
            WekaForecaster forecaster = new WekaForecaster();
            List<String> overlayFields = new ArrayList<String>();
           /* myHashMap hashMap = new myHashMap();
            for (int i = 1; i < 1392 ; i+=48) {
                hashMap.fillUpHashMap(applyFilterClassifier.applyFilterClassifier(data, i, i+47), 4, hashMap, data.attribute(1).name());
            }
            myHashMap.sortHashMapByValues(hashMap);
            String chosenLags = myHashMap.printHashMapFeatures(hashMap, 75);*/
            /*for (int i = 0; i < data.numAttributes()-2; i++)                                        //first 2 attributes are time and field to lag
                overlayFields.add(i, data.attribute(i+2).name());
            forecaster.getTSLagMaker().setOverlayFields(overlayFields);
            forecaster.getTSLagMaker().setTimeStampField(data.attribute(0).name()); // date time stamp
            forecaster.setFieldsToForecast(data.attribute(1).name());
            forecaster.setBaseForecaster(classifier);
            forecaster.getTSLagMaker().setIncludePowersOfTime(true);
            forecaster.getTSLagMaker().setIncludeTimeLagProducts(false);
            //forecaster.getTSLagMaker().setFieldsToLagAsString(data.attribute(1).name() + ", " + data.attribute(2).name());
            forecaster.getTSLagMaker().setMinLag(1);
            forecaster.getTSLagMaker().setMaxLag(100);
            //forecaster.getTSLagMaker().setLagRange("1, 2, 4, 8, 12, 96, 672, 20, 576, 384, 480, 192");
            crossValidateTS(data, forecaster);
            calculateErrors(true,  "MAPE");*/

            TSLagMaker tsLagMaker = new TSLagMaker();
            tsLagMaker.setFieldsToLagAsString(data.attribute(1).name());
            tsLagMaker.setTimeStampField(data.attribute(0).name());
            tsLagMaker.setIncludePowersOfTime(true);
            tsLagMaker.setIncludeTimeLagProducts(false);
            tsLagMaker.setMinLag(1);
            tsLagMaker.setMaxLag(1430);
            //tsLagMaker.setLagRange(chosenLags);
            tsLagMaker.setLagRange("768, 1, 769, 2, 3, 4, 1291, 1292, 527, 528, 1296, 1049, 282, 1051, 286, 287, 1055, 288, 1056, 289, 290, 1058, 814, 815, 816, 817, 573, 1341, 574, 1342, 575, 1343, 576, 1344, 577, 578, 579, 1102, 335, 1103, 336, 1104, 93, 94, 862, 95, 863, 96, 864, 97, 865, 98, 99, 101, 1389, 1390, 1391, 624, 1392, 381, 1149, 382, 1150, 383, 1151, 384, 1152, 385, 910, 911, 912, 914, 668, 669, 671");
           // tsLagMaker.setRemoveLeadingInstancesWithUnknownLagValues(true);
            //tsLagMaker.setLagRange("768, 1, 769, 2, 3, 4, 527, 528,  282, 286, 287, 288, 289, 290, 573, 574,  575, 576,  577, 578, 579, 335, 336,  93, 94, 95, 96, 97, 98, 99, 101, 624,  381, 382, 383, 384, 385,  668, 669, 671");

            for (int i = 0; i < data.numAttributes()-2; i++)                                        //first 2 attributes are time and field to lag
                overlayFields.add(i, data.attribute(i+2).name());
            tsLagMaker.setOverlayFields(overlayFields);
            Instances laggedData = tsLagMaker.getTransformedData(data);
           src.BestFirst bestFirst = new src.BestFirst();
           //bestFirst.setOptions(weka.core.Utils.splitOptions("-D 2"));
           SimmulatedAnnealing simmulatedAnnealing = new SimmulatedAnnealing();

           simmulatedAnnealing.search(laggedData, tsLagMaker, overlayFields);
           // bestFirst.search(laggedData, tsLagMaker, overlayFields);
       /*forecaster.setTSLagMaker(tsLagMaker);
            forecaster.setFieldsToForecast(data.attribute(1).name());
            tsLagMaker.setLagRange("3, 93, 94, 95, 97, 282, 287, 289, 290, 335, 381, 383, 384, 385, 573, 668, 669, 671, 768, 769, 814, 816, 817, 862, 863, 864, 910, 914, 1049, 1056, 1058, 1104, 1150, 1151, 1152, 1342, 1389, 1390, 1391");
            crossValidateTS(data, forecaster);
            resultLog.println(forecaster);
            calculateErrors(true, "MAPE", resultLog);*/

            long stopTime = System.currentTimeMillis();
            double elapsedTime = ((double) stopTime - startTime)/1000;
            resultLog.println("Time taken: " + elapsedTime);
            resultLog.close();
        }catch (Exception e){
            e.printStackTrace();
        }
    }
    public void crossValidateTS(Instances data, weka.classifiers.timeseries.WekaForecaster forecaster){
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
                if(!forecaster.getTSLagMaker().getOverlayFields().isEmpty())                        //checking if any overlay fields are set
                    forecast = forecaster.forecast(stepNumber, testData);
                else
                    forecast = forecaster.forecast(stepNumber);
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
    public double calculateErrors (boolean printOutput, String evaluationMeasure, PrintWriter resultLog){
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
                resultLog.println(errorOutput);
        }
        if(evaluationMeasure == "RMSE")
            getLastError = Math.sqrt(squaredErrorSum/ (i + 1));
        else if (evaluationMeasure == "MAPE")
            getLastError = piErrorSum/(i+1);
        return getLastError;
    }

    public Float getAvg(Float[] array){
        Float avg = (float) 0;
        for (int i = 0; i< array.length; i++)
            avg += array[i];
        return avg/array.length;


    }
    public void resetOptions(){
        actualValuesList = null;
        forecastedValuesList = null;
        map = null;
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