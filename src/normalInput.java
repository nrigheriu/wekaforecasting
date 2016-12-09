import java.io.*;

import java.text.DecimalFormat;
import java.util.*;
import javax.swing.JPanel;

import weka.attributeSelection.*;
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
import weka.classifiers.functions.SMOreg;
import weka.classifiers.functions.MLPRegressor;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.Utils.*;
import weka.core.Option;
import weka.filters.Filter;
import weka.filters.supervised.attribute.TSLagMaker;
import weka.attributeSelection.ASSearch;
import weka.core.Option;
public class normalInput {

    public static void main(String[] args) {
        try {
            String pathToAirlineData = "/home/cycle/workspace/airline.arff";
            String pathToTrainData = "/home/cycle/workspace/15_min_train.arff";
            String pathToTestData = "/home/cycle/workspace/15_min_test.arff";
            PrintWriter resultLog = new PrintWriter(new FileWriter("/home/cycle/workspace/wekaforecasting/results.txt", true));
            // load the data
            Instances trainData = new Instances(new BufferedReader(new FileReader(pathToTrainData)));
            Instances testData = new Instances(new BufferedReader(new FileReader(pathToTestData)));
            Instances airlineData = new Instances(new BufferedReader(new FileReader(pathToAirlineData)));

            airlineData.setClassIndex(airlineData.numAttributes()-2);
            trainData.setClassIndex(trainData.numAttributes()-1);
            long startTime = System.currentTimeMillis();

            //select_Attributes(airlineData);
            AttributeSelectedClassifier attributeSelectedClassifier = applyMetaClassifier(airlineData);
            AttributeSelectedClassifier attributeSelectedClassifier2 = new AttributeSelectedClassifier();

            doForecasting(trainData, resultLog, attributeSelectedClassifier, testData);
            long stopTime = System.currentTimeMillis();
            long elapsedTime = stopTime - startTime;
            resultLog.println("Time taken: " + elapsedTime);
            resultLog.close();
/*            TSEvaluation evaluation = new TSEvaluation(airlineData, stepNumber);
            evaluation.setHorizon(stepNumber);
            evaluation.evaluateForecaster(forecaster);
            System.out.println(evaluation.toSummaryString());*/
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
    public static void buildErrorGraph(Instances data, WekaForecaster forecaster,List<List<NumericPrediction>> forecast, int stepNumber){
        JFreeChartDriver graph = new JFreeChartDriver();
        String[] targetNames = new String[1];
        targetNames[0] = "sum";
//        List<Integer> steps = new ArrayList<>();
//        steps.add(100);
        ErrorModule errorModule = new ErrorModule();
        errorModule.setTargetFields(Arrays.asList(targetNames));
        List<ErrorModule> errorModuleList = new ArrayList<>();
        try {
            for (int i = 0; i< stepNumber; i++){
                Instance instance = data.get(i);
                errorModule.evaluateForInstance(forecast.get(i), instance);
                System.out.println(errorModule.toSummaryString());
                errorModuleList.add(errorModule);
            }
            JPanel panel2 = graph.getGraphPanelTargets(forecaster, errorModule, Arrays.asList(targetNames), 0, 1, data);
            graph.saveChartToFile(panel2, "file1", 1000, 1000);
        }catch (Exception e){
            e.printStackTrace();
        }
    }
    public static Instances select_Attributes(Instances data){
        weka.filters.supervised.attribute.AttributeSelection filter = new weka.filters.supervised.attribute.AttributeSelection();
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
    public static AttributeSelectedClassifier applyMetaClassifier(Instances trainData) {
        try {
            AttributeSelectedClassifier attributeSelectedClassifier = new AttributeSelectedClassifier();
            WrapperSubsetEval wrapperSubsetEval = new WrapperSubsetEval();
            //CfsSubsetEval cfsSubsetEval = new CfsSubsetEval();
            //HoltWinters holtWinters = new HoltWinters();
            ReliefFAttributeEval reliefFAttributeEval = new ReliefFAttributeEval();
            //PrincipalComponents principalComponents = new PrincipalComponents();
            SMOreg smOreg = new SMOreg();
            Ranker ranker = new Ranker();
            LinearRegression linearRegression = new LinearRegression();
            MultilayerPerceptron multilayerPerceptron = new MultilayerPerceptron();
            MLPRegressor mlpRegressor = new MLPRegressor();
            //mlpRegressor.buildClassifier(trainData);
            //SymmetricalUncertAttributeEval symmetricalUncertAttributeEval = new SymmetricalUncertAttributeEval();
            //attributeSelectedClassifier.setOptions(new String[]{"-D"});
            linearRegression.setOptions(weka.core.Utils.splitOptions("-S 1"));
            //linearRegression.buildClassifier(trainData);
            //wrapperSubsetEval.setOptions(weka.core.Utils.splitOptions("-F 0 -B weka.classifiers.functions.MLPRegressor"));

            //wrapperSubsetEval.setClassifier(linearRegression);
            //wrapperSubsetEval.buildEvaluator(trainData);
            BestFirst bestFirstsearch = new BestFirst();
            //search.setSearchBackwards(true);
            attributeSelectedClassifier.setSearch(ranker);
            attributeSelectedClassifier.setEvaluator(reliefFAttributeEval);
            attributeSelectedClassifier.setClassifier(linearRegression);

            //attributeSelectedClassifier.buildClassifier(trainData);
            return attributeSelectedClassifier;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }
    public static void doForecasting(Instances data, PrintWriter resultLog, AttributeSelectedClassifier attributeSelectedClassifier, Instances testData){
        try {
            WekaForecaster forecaster = new WekaForecaster();
            forecaster.setFieldsToForecast("sum");
            //select_Attributes(matrixData);
            //forecaster.setOptions(weka.core.Utils.splitOptions("-F passenger_numbers -G Date -quarter -month -W \"weka.classifiers.meta.AttributeSelectedClassifier " + "-W weka.classifiers.functions.LinearRegression\" -prime 12\""));
            //LinearRegression linearRegression = new LinearRegression();
            //MultilayerPerceptron multilayerPerceptron = new MultilayerPerceptron();
            //linearRegression.setOptions(weka.core.Utils.splitOptions("-S 1"));
            forecaster.setBaseForecaster(attributeSelectedClassifier);
            forecaster.getTSLagMaker().setTimeStampField("local_15min"); // date time stamp
            forecaster.getTSLagMaker().setMinLag(1);
            forecaster.getTSLagMaker().setMaxLag(360);
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

            int stepNumber = 12;
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
                System.out.println(errorOutput);
                resultLog.println(errorOutput);
            }
            resultLog.println("\n");
            buildErrorGraph(testData, forecaster, forecast, stepNumber);
        }catch (Exception e){
            e.printStackTrace();
        }
    }
    public static void returnSelectedAttributes(){
             /*AttributeSelection attsel  =new AttributeSelection();
            CfsSubsetEval cfsSubsetEval = new CfsSubsetEval();
            GreedyStepwise search = new GreedyStepwise();
            search.setSearchBackwards(true);
            attsel.setEvaluator(cfsSubsetEval);
            attsel.setSearch(search);
            attsel.SelectAttributes(airlineData);
            int[] indices = attsel.selectedAttributes();
            System.out.println("selected attribute :\n" + Utils.arrayToString(indices));*/
    }
}