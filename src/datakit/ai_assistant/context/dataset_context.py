"""
Dataset context generation module.

Creates a compact representation of the dataset
for the AI assistant.
"""


import pandas as pd



class DatasetContextBuilder:
    """
    Generate dataset information
    usable by the RAG pipeline.
    """



    def __init__(
        self,
        max_categories=10
    ):

        self.max_categories = max_categories



    def build(
        self,
        dataframe: pd.DataFrame,
        dataset_name="unknown"
    ):
        """
        Build dataset context.
        """


        context = {

            "dataset_name":
                dataset_name,

            "shape":
                {
                    "rows":
                        dataframe.shape[0],

                    "columns":
                        dataframe.shape[1]
                },


            "columns":
                self._analyze_columns(dataframe),


            "quality":
                self._analyze_quality(dataframe)

        }


        return context




    def _analyze_columns(
        self,
        dataframe
    ):

        columns = []


        for column in dataframe.columns:


            series = dataframe[column]


            info = {

                "name":
                    column,


                "dtype":
                    str(series.dtype),


                "missing_count":
                    int(series.isna().sum()),


                "missing_percentage":
                    round(
                        series.isna()
                        .mean()
                        * 100,
                        2
                    ),


                "unique_values":
                    int(
                        series.nunique()
                    )

            }



            if series.dtype == "object":

                info["top_values"] = (
                    series
                    .value_counts()
                    .head(
                        self.max_categories
                    )
                    .to_dict()
                )



            if pd.api.types.is_numeric_dtype(series):

                info["statistics"] = {

                    "mean":
                        float(
                            series.mean()
                        )
                        if not series.isna().all()
                        else None,


                    "median":
                        float(
                            series.median()
                        )
                        if not series.isna().all()
                        else None,


                    "std":
                        float(
                            series.std()
                        )
                        if not series.isna().all()
                        else None
                }


            columns.append(info)


        return columns





    def _analyze_quality(
        self,
        dataframe
    ):

        return {

            "duplicates":
                int(
                    dataframe
                    .duplicated()
                    .sum()
                ),


            "total_missing_values":
                int(
                    dataframe
                    .isna()
                    .sum()
                    .sum()
                )
        }