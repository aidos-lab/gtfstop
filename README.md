# gtfstop
Topological Analysis of GTFS data

To generate transit graphs from GTFS data the following steps were used - 
1. Firstly, download the relevant GTFS data, typically available in zip files. For this project, transit feeds from Germany have been used [taken from - https://gtfs.de/en/main/]
2. Follow the instructions provided in the colab notebook GTFS_Graphs.ipynb to generate the transit graph. For example, a graph of the routes covered by the ICE was plotted from the given feed.
3. Huge shoutout to Kuan for creating peartree [https://github.com/kuanb/peartree] that helps in converting GTFS feeds to networkX graphs using partridge objects! 
