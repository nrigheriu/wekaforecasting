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
import java.util.ArrayList;
import java.util.List;

/**
 * Created by cycle on 09.12.16.
 */
public class doForecasting {

    public static void doForecasting(Instances data, PrintWriter resultLog, Classifier classifier, Instances testData){
        try {
            long startTime = System.currentTimeMillis();
            WekaForecaster forecaster = new WekaForecaster();
            forecaster.setFieldsToForecast(data.attribute(1).name());
            //forecaster.setOptions(weka.core.Utils.splitOptions("-F passenger_numbers -G Date -quarter -month -W \"weka.classifiers.meta.AttributeSelectedClassifier " + "-W weka.classifiers.functions.LinearRegression\" -prime 12\""));
            forecaster.setBaseForecaster(classifier);
            forecaster.getTSLagMaker().setTimeStampField(data.attribute(0).name()); // date time stamp
            forecaster.getTSLagMaker().setMinLag(1);
            forecaster.getTSLagMaker().setMaxLag(100);
            //forecaster.setOptions(weka.core.Utils.splitOptions("-trim-leading"));
            //forecaster.getTSLagMaker().setAddDayOfWeek(true);
            //forecaster.getTSLagMaker().setIncludePowersOfTime(false);
            //forecaster.getTSLagMaker().setIncludeTimeLagProducts(false);
           /* forecaster.getTSLagMaker().setLagRange("100, 99, 98, 97, 96, 95, 94, 93, " +
                    "   92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76,  1, 75, 74, 73, 72, 71, 70, 69, 68, 64, 63, 66, 67, 65, 62");*/
            forecaster.getTSLagMaker().setLagRange("50, 49, 48,  1, 47, 34, 35, 33, 46, 32, 36, 45, 37, 31, 38, 39, 30, 44, 29, 40," +
                    "100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 81, 82, 80, 90, 79, 83, 78, 84, 85, 88");
            //List<String> laggedAttributes = new ArrayList<>();
            //laggedAttributes.add("passenger_numbers");
            //forecaster.getTSLagMaker().setFieldsToLag(laggedAttributes);
            //forecaster.getTSLagMaker().setAddMonthOfYear(true);
            forecaster.buildForecaster(data, System.out);
            //System.out.println(((AttributeSelectedClassifier)forecaster.getBaseForecaster()).measureNumAttributesSelected());
            forecaster.primeForecaster(data);
            int stepNumber = 24;
            List<List<NumericPrediction>> forecast = forecaster.forecast(stepNumber, System.out);
            //resultLog.println("AttrselClas built before!! " + forecaster.getBaseForecaster());
            //resultLog.println(forecaster.getAlgorithmName());
            //resultLog.println(forecaster.getBaseForecaster());
            System.out.println(getRankedList(forecaster, 20));
            resultLog.println(forecaster);
            double errorSum = 0;
            double piErrorSum = 0;
            DecimalFormat df = new DecimalFormat("#.###");
            List<String> errorList = new ArrayList<>();
            for (int i = 0; i < stepNumber; i++) {
                //double actualValue = testData.get(testData.size() - stepNumber + i).value(1);
                double actualValue = testData.get(i).value(1);
                double error = Math.abs(forecast.get(i).get(0).predicted() - actualValue);
                double piError = 100 * error / actualValue;
                piErrorSum += piError;
                errorSum += error;
                String errorOutput = "Step: " + i + " Prediction:" + df.format(forecast.get(i).get(0).predicted()) +
                        " Actual: " + actualValue +
                        " MAE: " + df.format(errorSum / (i + 1)) + " RMSE:" + df.format(Math.sqrt(Math.pow(error, 2)) / (i + 1)) +
                        " MAPE:" + df.format(piErrorSum / (i + 1));
                resultLog.println(errorOutput);
            }
            long stopTime = System.currentTimeMillis();
            long elapsedTime = stopTime - startTime;
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

            buildErrorGraph.buildErrorGraph(testData, forecaster, forecast, stepNumber);
        }catch (Exception e){
            e.printStackTrace();
        }
    }
    public static String getRankedList(WekaForecaster forecaster, int featureNumber){
        String forecasterResult = String.valueOf(forecaster);
        String features = new String();
        String lag_no = new String();
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
                        if (line.contains("Lag_sum")) {
                            lag_no += line.substring(line.length() - 3, line.length()).replaceAll("[^0-9]", " ");
                            if (!features.contains(lag_no)) {
                                features += lag_no + ",";
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
            features = features.substring(0, features.length() - 1);
            return features;
        }
        return null;
    }
}