package src;

import com.sun.org.apache.xpath.internal.SourceTree;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.NumericPrediction;
import weka.classifiers.timeseries.WekaForecaster;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.io.StringReader;
import java.text.DecimalFormat;
import java.util.*;

/**
 * Created by cycle on 09.12.16.
 */
public class doForecasting {
    static List<Double> actualValuesList = new ArrayList<>();
    static List<Double> forecastedValuesList = new ArrayList<>();

    public static void doForecasting(Instances data, PrintWriter resultLog, Classifier classifier){
        try {
            long startTime = System.currentTimeMillis();


            WekaForecaster forecaster = new WekaForecaster();
            forecaster.setFieldsToForecast(data.attribute(1).name());
            forecaster.setBaseForecaster(classifier);
            forecaster.getTSLagMaker().setTimeStampField(data.attribute(0).name()); // date time stamp
            forecaster.getTSLagMaker().setMinLag(1);
            forecaster.getTSLagMaker().setMaxLag(150);
            //forecaster.getTSLagMaker().setLagRange("579, 578, 580, 581, 577, 582, 576, 583, 575, 584, 574, 585, 586, 587, 573, 588, 496, 589, 572, 590, 591, 497, 592, 571, 593, 498, 594, 595, 570, 499, 596, 660, 500, 569, 597, 501, 659, 568, 598, 502, 567, 658, 630, 599, 503, 632, 631, 629, 633, 628, 634, 600, 537, 536, 535, 504, 566, 657, 538, 627, 539, 534, 635, 533, 541, 540, 626, 601, 248, 542, 505, 249, 532, 250, 247, 565, 251, 246, 636, 543, 252, 625, 245, 656, 531, 253, 602, 544, 530, 506");
            //forecaster.setOptions(weka.core.Utils.splitOptions("trim-leading"));
            //forecaster.getTSLagMaker().setAddDayOfWeek(true);
            forecaster.getTSLagMaker().setIncludePowersOfTime(false);
            //forecaster.getTSLagMaker().setIncludeTimeLagProducts(false);
            //forecaster.getTSLagMaker().setFieldsToLag(laggedAttributes);
            //forecaster.getTSLagMaker().setAddMonthOfYear(true);
            crossValidateTS(data, forecaster);



            int featureNumber = 15;
            HashMap<Integer, Float> map = new HashMap<Integer, Float>();
            map = getRankedList(forecaster, featureNumber, map);
           /* //map = sortHashMapByValues(map);
            //second lag part
            WekaForecaster forecaster2 = new WekaForecaster();
            forecaster2.setFieldsToForecast(data.attribute(1).name());
            forecaster2.setBaseForecaster(classifier);
            forecaster2.getTSLagMaker().setTimeStampField(data.attribute(0).name()); // date time stamp
            forecaster2.getTSLagMaker().setMinLag(166);
            forecaster2.getTSLagMaker().setMaxLag(330);
            forecaster2.buildForecaster(data, System.out);
            map = getRankedList(forecaster2, 90, map);



            WekaForecaster forecaster3 = new WekaForecaster();
            forecaster3.setFieldsToForecast(data.attribute(1).name());
            forecaster3.setBaseForecaster(classifier);
            forecaster3.getTSLagMaker().setTimeStampField(data.attribute(0).name()); // date time stamp
            forecaster3.getTSLagMaker().setMinLag(331);
            forecaster3.getTSLagMaker().setMaxLag(495);
            forecaster3.buildForecaster(data, System.out);
            map = getRankedList(forecaster3, 90, map);

            WekaForecaster forecaster4 = new WekaForecaster();
            forecaster4.setFieldsToForecast(data.attribute(1).name());
            forecaster4.setBaseForecaster(classifier);
            forecaster4.getTSLagMaker().setTimeStampField(data.attribute(0).name()); // date time stamp
            forecaster4.getTSLagMaker().setMinLag(496);
            forecaster4.getTSLagMaker().setMaxLag(660);
            forecaster4.buildForecaster(data, System.out);
            map = getRankedList(forecaster4, 90, map);*/

            /*map = sortHashMapByValues(map);
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


            resultLog.println(forecaster);
            calculateErrors(resultLog, actualValuesList, forecastedValuesList);

            long stopTime = System.currentTimeMillis();
            double elapsedTime = ((double) stopTime - startTime)/1000;
            resultLog.println("Time taken: " + elapsedTime);
            resultLog.close();
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


        }catch (Exception e){
            e.printStackTrace();
        }
    }
    public static void crossValidateTS(Instances data, WekaForecaster forecaster){
        try {
            int stepNumber = 24;
            Instances testData = null, trainData = null;
            List<List<NumericPrediction>> forecast = null;
            for (int trainingPercentage = 70; trainingPercentage <= 95; trainingPercentage += 5) {
                trainData = getSplittedData(data, trainingPercentage, true);
                testData = getSplittedData(data, trainingPercentage, false);
                forecaster.buildForecaster(trainData, System.out);
                forecaster.primeForecaster(trainData);
                forecast = forecaster.forecast(stepNumber, System.out);
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
    public static void calculateErrors (PrintWriter resultLog, List<Double> actualValuesList, List<Double> forecastedValuesList){
        double errorSum = 0;
        double piErrorSum = 0;
        DecimalFormat df = new DecimalFormat("#.###");
        List<String> errorList = new ArrayList<>();
        for (int i = 0; i < actualValuesList.size(); i++) {
            //double actualValue = testData.get(testData.size() - stepNumber + i).value(1);
            double actualValue = actualValuesList.get(i);
            double error = Math.abs(forecastedValuesList.get(i) - actualValue);
            double piError = 100 * error / actualValue;
            piErrorSum += piError;
            errorSum += error;
            String errorOutput = "Step: " + i + " Prediction:" + df.format(forecastedValuesList.get(i)) +
                    " Actual: " + actualValue +
                    " MAE: " + df.format(errorSum / (i + 1)) + " RMSE:" + df.format(Math.sqrt(Math.pow(error, 2)) / (i + 1)) +
                    " MAPE:" + df.format(piErrorSum / (i + 1));
            resultLog.println(errorOutput);
        }
    }
    public static LinkedHashMap<Integer, Float> sortHashMapByValues(HashMap<Integer, Float> passedMap) {
        List<Integer> mapKeys = new ArrayList<>(passedMap.keySet());
        List<Float> mapValues = new ArrayList<>(passedMap.values());
        Collections.sort(mapValues);
        Collections.sort(mapKeys);
        Collections.reverse(mapKeys);
        Collections.reverse(mapValues);
        LinkedHashMap<Integer, Float> sortedMap = new LinkedHashMap<>();
        Iterator<Float> valueIt = mapValues.iterator();
        while (valueIt.hasNext()) {
            Float val = valueIt.next();
            Iterator<Integer> keyIt = mapKeys.iterator();
            while (keyIt.hasNext()) {
                Integer key = keyIt.next();
                Float comp1 = passedMap.get(key);
                Float comp2 = val;
                if (comp1.equals(comp2)) {
                    keyIt.remove();
                    sortedMap.put(key, val);
                    break;
                }
            }
        }
        return sortedMap;
    }
    public static HashMap<Integer, Float> getRankedList(WekaForecaster forecaster, int featureNumber, HashMap<Integer, Float> map){
        String forecasterResult = String.valueOf(forecaster);
        String features = new String();
        String lag_no = new String();
        String lagRankingVal = new String();

        int lineWhereRankingStarts = forecasterResult.indexOf("Ranked attributes:");
        if(lineWhereRankingStarts >= 0) {
            BufferedReader rankedPart = new BufferedReader(
                    new StringReader(forecasterResult.substring(lineWhereRankingStarts + 19, forecasterResult.length())));
            String line = null;
            int i = 0;
            while (i < featureNumber) {
                try {
                    line = rankedPart.readLine();
                    if (line != null && !line.contains("Selected attributes:")) {
                        lag_no = "";
                        lagRankingVal = "";
                        if (line.contains("Lag_sum")) {
                            lagRankingVal += line.substring(0, 10);
                            lag_no += line.substring(line.length() - 3, line.length()).replaceAll("[^0-9]", " ");
                            lag_no = lag_no.replaceAll("\\s", "");
                            if (!features.contains(lag_no)) {
                                features += lag_no + ",";
                                map.put(Integer.valueOf(lag_no), Float.valueOf(lagRankingVal));
                                i++;
                            }
                        }
                    } else {
                        break;
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                    break;
                }
            }
            //System.out.println(features);
            return map;
        }
        return null;
    }
}
