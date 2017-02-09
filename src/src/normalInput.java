package src;

import java.io.*;

import java.text.DecimalFormat;
import java.util.*;
import javax.swing.JPanel;

import weka.attributeSelection.*;
import weka.classifiers.functions.*;
import weka.classifiers.lazy.IBk;
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
            String pathToWholeData = "/home/cycle/workspace/wekaforecasting-new-features/dataSets/3months_100aggregate_extraFeatures.arff";
            String pathToHugeData = "/home/cycle/workspace/wekaforecasting-new-features/dataSets/6months_1aggregate_extraFeatures.arff";

            // load the data
            Instances wholeData  = new Instances(new BufferedReader(new FileReader(pathToWholeData)));
            Instances hugeData = new Instances(new BufferedReader((new FileReader(pathToHugeData))));

            wholeData.setClassIndex(1);
            hugeData.setClassIndex(1);
            //select_Attributes(airlineData);
            //AttributeSelectedClassifier attributeSelectedClassifier = applyMetaClassifier.applyMetaClassifier(airlineData);
            AttributeSelectedClassifier attributeSelectedClassifier2 = new AttributeSelectedClassifier();
            IBk iBk = new IBk();
            LinearRegression linearRegression = new LinearRegression();
            linearRegression.setOptions(weka.core.Utils.splitOptions("-S 1 -R 1E-6"));
            MLPRegressor mlpRegressor = new MLPRegressor();
            //mlpRegressor.setOptions(weka.core.Utils.splitOptions("-N 1"));
            doForecasting doForecasting = new doForecasting();
            doForecasting.doForecast(wholeData, linearRegression);
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
}
