package src;

import java.io.*;

import java.text.DecimalFormat;
import java.util.*;
import javax.swing.JPanel;

import weka.attributeSelection.*;
import weka.classifiers.functions.*;
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
            String pathPracticeData = "/home/cycle/workspace/tsdata.arff";
            PrintWriter resultLog = new PrintWriter(new FileWriter("/home/cycle/workspace/wekaforecasting/new_results.txt", true));

            // load the data
            Instances trainData = new Instances(new BufferedReader(new FileReader(pathToTrainData)));
            Instances testData = new Instances(new BufferedReader(new FileReader(pathToTestData)));
            Instances airlineData = new Instances(new BufferedReader(new FileReader(pathToAirlineData)));
            Instances practiceData = new Instances(new BufferedReader(new FileReader(pathPracticeData)));
            Instances trainPractice = new Instances(practiceData, 0, 5);
            Instances testPractice = new Instances(practiceData, trainPractice.size(), practiceData.size()-trainPractice.size());

            airlineData.setClassIndex(airlineData.numAttributes()-2);
            trainData.setClassIndex(trainData.numAttributes()-1);
            trainPractice.setClassIndex(practiceData.numAttributes()-1);
            //select_Attributes(airlineData);
            //AttributeSelectedClassifier attributeSelectedClassifier = applyMetaClassifier.applyMetaClassifier(airlineData);
            AttributeSelectedClassifier attributeSelectedClassifier2 = new AttributeSelectedClassifier();

            LinearRegression linearRegression = new LinearRegression();
            linearRegression.setOptions(weka.core.Utils.splitOptions("-S 1"));
            MLPRegressor mlpRegressor = new MLPRegressor();
            mlpRegressor.setOptions(weka.core.Utils.splitOptions("-N 1"));
            doForecasting.doForecasting(trainData, resultLog, mlpRegressor, testData);
        } catch (Exception ex) {
            ex.printStackTrace();
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
