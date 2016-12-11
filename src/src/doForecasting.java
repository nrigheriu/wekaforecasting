package src;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.NumericPrediction;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.timeseries.WekaForecaster;
import weka.core.Instances;
import src.lagInput;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by cycle on 09.12.16.
 */
public class doForecasting {

    public static void doForecasting(Instances data, PrintWriter resultLog, Classifier classifier, Instances testData){
        try {
            WekaForecaster forecaster = new WekaForecaster();
            forecaster.setFieldsToForecast(data.attribute(1).name());
            //select_Attributes(matrixData);
            //forecaster.setOptions(weka.core.Utils.splitOptions("-F passenger_numbers -G Date -quarter -month -W \"weka.classifiers.meta.AttributeSelectedClassifier " + "-W weka.classifiers.functions.LinearRegression\" -prime 12\""));
            //LinearRegression linearRegression = new LinearRegression();
            //MultilayerPerceptron multilayerPerceptron = new MultilayerPerceptron();
            //linearRegression.setOptions(weka.core.Utils.splitOptions("-S 1"));
            forecaster.setBaseForecaster(classifier);
            forecaster.getTSLagMaker().setTimeStampField(data.attribute(0).name()); // date time stamp
            forecaster.getTSLagMaker().setMinLag(0);
            forecaster.getTSLagMaker().setMaxLag(4);
            forecaster.getTSLagMaker().setIncludePowersOfTime(false);
            forecaster.getTSLagMaker().setIncludeTimeLagProducts(false);
            forecaster.getTSLagMaker().setLagRange("1");
            //forecaster.getTSLagMaker().setLagRange("1-3, 6, 9, 12, 14, 20, 23, 35, 36, 46, 52, 58, 84, 93, 94, 96");
            //forecaster.getTSLagMaker().setLagRange("96-74, 60-64, 1-5, 57-59");
            //List<String> laggedAttributes = new ArrayList<>();
            //laggedAttributes.add("passenger_numbers");
            //forecaster.getTSLagMaker().setFieldsToLag(laggedAttributes);
            //forecaster.getTSLagMaker().setAddMonthOfYear(true);
            //forecaster.getTSLagMaker().setAddQuarterOfYear(true);

            forecaster.buildForecaster(data, System.out);

            //System.out.println(((AttributeSelectedClassifier)forecaster.getBaseForecaster()).measureNumAttributesSelected());
            forecaster.primeForecaster(data);

            int stepNumber = 4;
            List<List<NumericPrediction>> forecast = forecaster.forecast(stepNumber, System.out);
            //resultLog.println("AttrselClas built before!! " + forecaster.getBaseForecaster());
            //resultLog.println(forecaster.getAlgorithmName());
            //resultLog.println(forecaster.getBaseForecaster());
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
            //resultLog.println(forecaster.getTSLagMaker().createTimeLagCrossProducts(data) + "\n");
            resultLog.println(forecaster.getTSLagMaker().getTransformedData(data) + "\n");

            buildErrorGraph.buildErrorGraph(testData, forecaster, forecast, stepNumber);
        }catch (Exception e){
            e.printStackTrace();
        }
    }
}
