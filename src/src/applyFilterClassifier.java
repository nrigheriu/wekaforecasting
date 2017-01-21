package src;

import weka.attributeSelection.Ranker;
import weka.attributeSelection.ReliefFAttributeEval;
import weka.core.Instances;
import weka.filters.supervised.attribute.TSLagMaker;

/**
 * Created by cycle on 07.01.2017.
 */
public class applyFilterClassifier {

    public applyFilterClassifier(){

    }

    public String applyFilterClassifier(Instances data, int minLag, int maxLag){
        try {
            weka.attributeSelection.AttributeSelection attsel = new weka.attributeSelection.AttributeSelection();
            TSLagMaker tsLagMaker = new TSLagMaker();
            tsLagMaker.setFieldsToLagAsString(data.attribute(1).name());
            tsLagMaker.setTimeStampField(data.attribute(0).name());
            tsLagMaker.setIncludePowersOfTime(true);
            tsLagMaker.setIncludeTimeLagProducts(false);
            tsLagMaker.setMinLag(minLag);
            tsLagMaker.setMaxLag(maxLag);
            Instances laggedData = tsLagMaker.getTransformedData(data);
            ReliefFAttributeEval reliefFAttributeEval = new ReliefFAttributeEval();
            Ranker ranker = new Ranker();
            attsel.setEvaluator(reliefFAttributeEval);
            attsel.setSearch(ranker);
            attsel.SelectAttributes(laggedData);
            int[] indices = attsel.selectedAttributes();
            System.out.println(attsel.toResultsString());
            return attsel.toResultsString();
        }catch (Exception e){
            e.printStackTrace();
        }
            return null;
    }
}
