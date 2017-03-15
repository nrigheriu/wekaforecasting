package src;

import java.io.BufferedReader;
import java.io.StringReader;
import java.util.*;
import weka.classifiers.timeseries.WekaForecaster;

/**
 * Created by cycle on 17.12.16.
 */
public class MyHashMap extends HashMap{
    HashMap<Integer, Float> map = new HashMap<Integer, Float>();
    public MyHashMap(){

    }

    public HashMap<Integer, Float> fillUpHashMap(String result, int featureNumber, String laggedFields){
        String features = new String();
        String lag_no = new String();
        String lagRankingVal = new String();

        int lineWhereRankingStarts = result.indexOf("Ranked attributes:");
        if(lineWhereRankingStarts >= 0) {
            BufferedReader rankedPart = new BufferedReader(
                    new StringReader(result.substring(lineWhereRankingStarts + 19, result.length())));
            String line = null;
            int i = 0;
            while (i < featureNumber) {
                try {
                    line = rankedPart.readLine();
                    if (line != null && !line.contains("Selected attributes:")) {
                        lag_no = "";
                        lagRankingVal = "";
                        if (line.contains("Lag_" + laggedFields)) {
                            lagRankingVal += line.substring(0, 8);
                            lag_no += line.substring(line.length() - 4, line.length()).replaceAll("[^0-9]", " ");
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
            return map;
        }
        return null;
    }
    public  LinkedHashMap<Integer, Float> sortHashMapByValues() {
        List<Integer> mapKeys = new ArrayList<>(this.map.keySet());
        List<Float> mapValues = new ArrayList<>(this.map.values());
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
                Float comp1 = this.map.get(key);
                Float comp2 = val;
                if (comp1.equals(comp2)) {
                    keyIt.remove();
                    sortedMap.put(key, val);
                    break;
                }
            }
        }
        map = sortedMap;
        return sortedMap;
    }
    public  String printHashMapFeatures(int featureNumber){
        Set<Integer> mapKeys = this.map.keySet();
        String combinedFeatures = "";
        String justLags = "";
        int j = 1;
        for(Integer key:mapKeys){
            if(j > featureNumber)
                break;
            combinedFeatures += String.valueOf(key) + " value: " + map.get(key) + "\n";
            justLags += String.valueOf(key) + ", ";
            j++;
        }
        System.out.println("HashMapFeatures: " + combinedFeatures);
        return justLags.substring(0, justLags.length()-2);
    }
}

