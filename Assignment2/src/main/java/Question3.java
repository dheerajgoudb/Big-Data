import java.io.IOException;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class Question3 {
	
	public static class TopNMapper extends Mapper<Object, Text, Text, IntWritable>
	{	
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException
		{
			String[] line = StringUtils.split(value.toString(), "^");
			String[] address = StringUtils.split(line[1], " ");
			
			context.write(new Text(address[address.length - 1]), new IntWritable(1));
		}
	}
	
	public static class TopNReducer extends Reducer<Text, IntWritable, Text, IntWritable>
	{
		private Map<Text, IntWritable> countMap = new HashMap<>();
		
		public void reduce(Text key, Iterable<IntWritable> values, Context context)
		{
			int sum = 0;
			for (IntWritable val : values)
			{
				sum += val.get();
			}
			countMap.put(new Text(key), new IntWritable(sum));
		}
		
		protected void cleanup(Context context) throws IOException, InterruptedException
		{
			Map<Text, IntWritable> sortedMap = sortByValues(countMap);
			
			int counter = 0;
			for (Text key : sortedMap.keySet())
			{
				if (counter++ == 10)
				{
					break;
				}
				context.write(key, sortedMap.get(key));
			}
		}
	}
	
	public static Map<Text, IntWritable> sortByValues(Map<Text, IntWritable> map) 
	{
		List<Map.Entry<Text, IntWritable>> entries = new LinkedList<Map.Entry<Text, IntWritable>>(map.entrySet());
		
		Collections.sort(entries, new Comparator<Map.Entry<Text, IntWritable>>() 
		{
			public int compare(Map.Entry<Text, IntWritable> o1, Map.Entry<Text, IntWritable> o2)
			{
				return o2.getValue().compareTo(o1.getValue());
			}
		});
		
		Map<Text, IntWritable> sortedMap = new LinkedHashMap<Text, IntWritable>();
		
		for (Map.Entry<Text, IntWritable> entry : entries)
		{
			sortedMap.put(entry.getKey(), entry.getValue());
		}
		return sortedMap;
	}

	public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException 
	{
		Configuration conf = new Configuration();
		String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
		if (otherArgs.length != 2)
		{
			System.err.println("Usage: Question3 <input file path> <output folder path>");
		}
		
		Job job = Job.getInstance(conf, "Question3");
		job.setJarByClass(Question3.class);
		job.setMapperClass(TopNMapper.class);
		job.setReducerClass(TopNReducer.class);
		
		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(IntWritable.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(IntWritable.class);
		
		FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
		FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
		
		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}

}
