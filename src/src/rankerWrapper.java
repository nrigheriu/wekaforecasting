package src;

import weka.core.Instances;
import weka.classifiers.timeseries.WekaForecaster;
import java.io.BufferedReader;
import java.io.PrintWriter;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * Created by cycle on 18.12.16.
 */
public class rankerWrapper {
    static int[] minLagValues;
    static int[] maxLagValues;
    static List<Double> actualValuesList = new ArrayList<>();
    static List<Double> forecastedValuesList = new ArrayList<>();
    static Float[] accuracy;
    static Float[] percentFeaturesToGetFromInterval;
    static ArrayList<ArrayList<Integer>> featureListForIntevals;
    static ArrayList<Integer> selectedFeatures;

    public rankerWrapper() {
        minLagValues = new int[]{1, 49, 97, 145, 193, 241, 289, 337, 385, 433, 481, 529, 577, 625, 673, 721};
        maxLagValues = new int[]{60, 108, 156, 204, 252, 300, 348, 396, 444, 492, 540, 588, 636, 684, 732, 780};
        actualValuesList = new ArrayList<>();
        forecastedValuesList = new ArrayList<>();
        accuracy = new Float[minLagValues.length];
        percentFeaturesToGetFromInterval = new Float[minLagValues.length];
        featureListForIntevals = new ArrayList<ArrayList<Integer>>(minLagValues.length);
        selectedFeatures = new ArrayList<Integer>();

    }
    public static Float[] getPercentagesForIntervals(WekaForecaster forecaster, Instances data, PrintWriter resultLog){
        actualValuesList.clear();
        forecastedValuesList.clear();
        for (int i = 0; i < minLagValues.length;i++){
            forecaster.getTSLagMaker().setMinLag(minLagValues[i]);
            forecaster.getTSLagMaker().setMaxLag(maxLagValues[i]);
            doForecasting.crossValidateTS(data, forecaster);
            accuracy[i] = doForecasting.calculateErrors(resultLog, false);
            System.out.println("Accuracy: " + accuracy[i]);
            ArrayList<Integer> intervalFeatureList = new ArrayList<Integer>();
            intervalFeatureList = getFeatureList(forecaster);
            featureListForIntevals.add(intervalFeatureList);
            //resultLog.println(forecaster);
            actualValuesList.clear();
            forecastedValuesList.clear();
        }
        Float avgRMSE = doForecasting.getAvg(accuracy);
        for (int i = 0; i < minLagValues.length;i++){
            percentFeaturesToGetFromInterval[i] = 100 + ((avgRMSE - accuracy[i])*100)/ avgRMSE;
            System.out.println("PercentfeaturesTo get From Interval, rounded " + i + " : " + Math.round(percentFeaturesToGetFromInterval[i]));

        }
        return percentFeaturesToGetFromInterval;
    }
    public int[] getMinLagValues(){
        return this.minLagValues;
    }
    public int[] getMaxLagValues(){
        return this.maxLagValues;
    }
    public void setMinLagValues(int[] minLagValues){
        this.minLagValues = minLagValues;
    }
    public void setMaxLagValues(int[] maxLagValues){
        this.maxLagValues = maxLagValues;
    }

    public static ArrayList<Integer> populateSelectedFeatures(ArrayList<ArrayList<Integer>> featureListForIntevals, Float[] percentFeaturesToGetFromInterval, int firstXfeatures){
        for(int i =0; i < featureListForIntevals.size(); i++){
            int featuresToGet = Math.round((percentFeaturesToGetFromInterval[i] * firstXfeatures)/100);
            System.out.println("Features to get from Interval: " + i + " " + featuresToGet);
            for (int j = 0; j < featuresToGet; j++) {
                selectedFeatures.add(featureListForIntevals.get(i).get(j));
            }
        }
        return selectedFeatures;
    }

    public static ArrayList<Integer> getFeatureList(WekaForecaster forecaster){
        String forecasterResult = String.valueOf(forecaster);
        String features = new String();
        String lag_no = new String();
        String lagRankingVal = new String();
        ArrayList<Integer> featureList = new ArrayList<>();
        int lineWhereRankingStarts = forecasterResult.indexOf("Ranked attributes:");
        if(lineWhereRankingStarts >= 0) {
            BufferedReader rankedPart = new BufferedReader(
                    new StringReader(forecasterResult.substring(lineWhereRankingStarts + 19, forecasterResult.length())));
            String line = null;
            int i = 0;
            while (true) {
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
                                featureList.add(Integer.valueOf(lag_no));
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
            return featureList;
        }
        return null;
    }
}
