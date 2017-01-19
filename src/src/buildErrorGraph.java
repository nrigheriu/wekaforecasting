package src;

import weka.classifiers.evaluation.NumericPrediction;
import weka.classifiers.timeseries.WekaForecaster;
import weka.classifiers.timeseries.eval.ErrorModule;
import weka.classifiers.timeseries.eval.graph.JFreeChartDriver;
import weka.core.Instance;
import weka.core.Instances;

import javax.swing.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by cycle on 09.12.16.
 */
public class buildErrorGraph {
    public static void buildErrorGraph(Instances testData, src.WekaForecaster forecaster, List<List<NumericPrediction>> forecast, int stepNumber){
        JFreeChartDriver graph = new JFreeChartDriver();
        String[] targetNames = new String[1];
        targetNames[0] = testData.attribute(1).name();
//        List<Integer> steps = new ArrayList<>();
//        steps.add(100);
        ErrorModule errorModule = new ErrorModule();
        errorModule.setTargetFields(Arrays.asList(targetNames));
        List<ErrorModule> errorModuleList = new ArrayList<>();
        try {
            for (int i = 0; i< stepNumber; i++){
                Instance instance = testData.get(i);
                errorModule.evaluateForInstance(forecast.get(i), instance);
                //System.out.println(errorModule.toSummaryString());
                errorModuleList.add(errorModule);
            }
            JPanel panel2 = graph.getGraphPanelTargets(forecaster, errorModule, Arrays.asList(targetNames), 0, 1, testData);
            graph.saveChartToFile(panel2, "file1", 1000, 1000);
        }catch (Exception e){
            e.printStackTrace();
        }
    }
}
