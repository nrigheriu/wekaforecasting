package src;

import java.io.*;

import java.text.DecimalFormat;
import java.util.*;
import javax.swing.JPanel;

import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.WrapperSubsetEval;
import weka.classifiers.functions.GaussianProcesses;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.meta.multisearch.RandomSearch;
import weka.classifiers.timeseries.HoltWinters;
import weka.classifiers.timeseries.TSForecaster;
import weka.classifiers.evaluation.NumericPrediction;
import weka.classifiers.timeseries.eval.ErrorModule;
import weka.classifiers.timeseries.WekaForecaster;
import weka.classifiers.timeseries.core.TSLagUser;
import weka.classifiers.timeseries.eval.TSEvaluation;
import weka.classifiers.timeseries.eval.graph.JFreeChartDriver;
import weka.classifiers.timeseries.eval.ErrorModule;
import weka.classifiers.timeseries.core.TSLagUser;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.core.*;
import weka.core.Utils.*;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.supervised.attribute.TSLagMaker;
import weka.attributeSelection.ASSearch;

public class lagInput {

    public static void main(String[] args) {
        try {
            String pathToAirlineData = "/home/cycle/workspace/airline.arff";
            String pathToTrainData = "/home/cycle/workspace/15_min_train_96lag.arff";
            String pathToTestData = "/home/cycle/workspace/15_min_test.arff";
            PrintWriter resultLog = new PrintWriter(new FileWriter("/home/cycle/workspace/wekaforecasting/results.txt", true));
            // load the data
            Instances trainData = new Instances(new BufferedReader(new FileReader(pathToTrainData)));
            Instances testData = new Instances(new BufferedReader(new FileReader(pathToTestData)));
            Instances airlineData = new Instances(new BufferedReader(new FileReader(pathToAirlineData)));

            //airlineData.setClassIndex(airlineData.numAttributes()-2);
            //matrixData.setClassIndex(matrixData.numAttributes()-1);
            trainData.setClassIndex(trainData.numAttributes()-1);
            //start Time
            long startTime = System.currentTimeMillis();
            AttributeSelectedClassifier attributeSelectedClassifier = applyMetaClassifier(trainData);
            //attributeSelectedClassifier.setOptions(new String[]{"-D"});
            // new forecaster
            WekaForecaster forecaster = new WekaForecaster();
            // set the targets we want to forecast. This method calls
            // setFieldsToLag() on the lag maker object for us
            forecaster.setFieldsToForecast("lagY");
            //selectAttributes(matrixData);
            forecaster.setBaseForecaster(attributeSelectedClassifier);
            forecaster.getTSLagMaker().setTimeStampField("local_15min"); // date time stamp
            //forecaster.getTSLagMaker().setMinLag(1);
            //forecaster.getTSLagMaker().setMaxLag(1); // monthly data
            List<String> laggedAttributes = new ArrayList<>();
            laggedAttributes.add("lagY");
            //forecaster.getTSLagMaker().setFieldsToLag(laggedAttributes);
            // add a month of the year indicator field
            //forecaster.getTSLagMaker().setAddMonthOfYear(true);
            // add a quarter of the year indicator field
            //forecaster.getTSLagMaker().setAddQuarterOfYear(true);
            // build the model
            forecaster.buildForecaster(trainData, System.out);
            //System.out.println(((AttributeSelectedClassifier)forecaster.getBaseForecaster()).measureNumAttributesSelected());
            forecaster.primeForecaster(trainData);
            long stopTime = System.currentTimeMillis();
            long elapsedTime = stopTime - startTime;
            resultLog.println("Time taken: " + elapsedTime);
            // forecast for 12 units (months) beyond the end of the training data
            int stepNumber = 8;
            List<List<NumericPrediction>> forecast = forecaster.forecast(stepNumber, System.out);
            // output the predictions. Outer list is over the steps; inner list is over the targets
            //resultLog.println("AttrselClas built before!! " + forecaster.getBaseForecaster());
            resultLog.println(forecaster.getAlgorithmName());
            resultLog.println(forecaster.getBaseForecaster());
            double errorSum = 0;
            double piErrorSum = 0;
            DecimalFormat df = new DecimalFormat("#.###");
            List<String> errorList = new ArrayList<>();
            for (int i = 0; i < stepNumber; i++) {
                double actualValue = testData.get(i).value(1);
                double error = Math.abs(forecast.get(i).get(0).predicted() - actualValue);
                double piError = 100*error/actualValue;
                piErrorSum += piError;
                errorSum += error;
                String errorOutput = "Step: " + i + " Prediction:"  + df.format(forecast.get(i).get(0).predicted())  +
                        " Actual: " + actualValue +
                        " MAE: " + df.format(errorSum/(i+1)) + " RMSE:" + df.format(Math.sqrt(Math.pow(error, 2))/(i+1)) +
                        " MAPE:" + df.format(piErrorSum/(i+1));
                System.out.println(errorOutput);
                resultLog.println(errorOutput);
            }
            resultLog.println("\n");
            resultLog.close();
/*            TSEvaluation evaluation = new TSEvaluation(airlineData, stepNumber);
            evaluation.setHorizon(stepNumber);
            evaluation.evaluateForecaster(forecaster);
            System.out.println(evaluation.toSummaryString());*/
            //plotting
            buildErrorGraph.buildErrorGraph(testData, forecaster, forecast, stepNumber);
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public static Instances selectAttributes(Instances data){
        AttributeSelection filter = new AttributeSelection();
        WrapperSubsetEval eval = new WrapperSubsetEval();
        CfsSubsetEval eval2 = new CfsSubsetEval();
        eval.setClassifier(new LinearRegression());
        GreedyStepwise search = new GreedyStepwise();
        search.setSearchBackwards(true);
        filter.setEvaluator(eval);
        filter.setSearch(search);
        try {
            filter.setInputFormat(data);
            Instances newData = Filter.useFilter(data, filter);
            System.out.println("new data:" + newData);
            return data;
        }catch (Exception e){
            e.printStackTrace();
        }
        return null;
    }
    public static AttributeSelectedClassifier applyMetaClassifier(Instances trainData){
        try {
            AttributeSelectedClassifier attributeSelectedClassifier = new AttributeSelectedClassifier();
            WrapperSubsetEval wrapperSubsetEval = new WrapperSubsetEval();
            //HoltWinters holtWinters = new HoltWinters();
            LinearRegression linearRegression = new LinearRegression();
            linearRegression.setOptions(weka.core.Utils.splitOptions("-S 0"));
            System.out.println(linearRegression.getAttributeSelectionMethod());
            linearRegression.buildClassifier(trainData);
            wrapperSubsetEval.setClassifier(linearRegression);
            BestFirst search = new BestFirst();
            //search.setSearchBackwards(true);
            attributeSelectedClassifier.setSearch(search);
            attributeSelectedClassifier.setEvaluator(wrapperSubsetEval);
            attributeSelectedClassifier.setClassifier(linearRegression);
            attributeSelectedClassifier.buildClassifier(trainData);
            return attributeSelectedClassifier;
        }catch (Exception e){
            e.printStackTrace();
        }
        return null;
    }
}