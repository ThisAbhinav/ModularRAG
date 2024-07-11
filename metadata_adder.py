import pandas as pd
from ast import literal_eval
import datetime

themedict = {
    "AB20.pdf": "Agri Bot",
    "AB23.pdf": "Astro Bot",
    "CL23.pdf": "Cosmo Logistic",
    "FR22.pdf": "Functional Weeder",
    "GG23.pdf": "Geo Guide",
    "HB23.pdf": "Hologlyph Bots",
    "LD23.pdf": "Luminosity Drone",
    "SB20.pdf": "Sahayak Bot",
    "SS21.pdf": "Vitarana Drone",
    "VB20.pdf": "Vargi Bots",
    "VD20.pdf": "Vitarana Drone",
}

yeardict = {
    "AB20.pdf": 2020,
    "AB23.pdf": 2023,
    "CL23.pdf": 2023,
    "FR22.pdf": 2022,
    "GG23.pdf": 2023,
    "HB23.pdf": 2023,
    "LD23.pdf": 2023,
    "SB20.pdf": 2020,
    "SS21.pdf": 2021,
    "VB20.pdf": 2020,
    "VD20.pdf": 2020,
}


def list_of_dict_function(l1):
    for x in l1:
        if "file_name" in x.keys():
            x["theme"] = themedict[x["file_name"]]
            x["year"] = yeardict[x["file_name"]]
        else:
            x["theme"] = themedict[x["filename"].split("/")[-1]]
            x["year"] = yeardict[x["filename"].split("/")[-1]]
    return l1


def add_metadata():
    df = pd.read_pickle("data/ragas_qa.pickle")

    # df["meta_data_real_contexts"] = df["meta_data_real_contexts"].apply(
    #     lambda x: (
    #         ",".join((x.split(",")[0:6] + x.split(",")[13:]))
    #         if "source" not in x
    #         else x
    #     )
    # )
    df["meta_data_real_contexts"] = df["meta_data_real_contexts"].apply(
        lambda x: literal_eval(str(x))
    )
    df["meta_data_real_contexts"] = df["meta_data_real_contexts"].apply(
        list_of_dict_function
    )

    df.to_pickle("data/ragas_qa.pickle")
    df.to_csv("data/ragas_qa.csv")


def list_of_dict_function2(l1):
    for x in l1:
        name = x["source"].split("/")[1] + ".pdf"
        x["theme"] = themedict[name]
        x["year"] = yeardict[name]
    return l1


def add_metadata_generated():
    df = pd.read_csv("results/pipeline_RecursiveChunking.csv")
    df["metadata"] = df["metadata"].apply(
        lambda l: list_of_dict_function2(literal_eval(str(l)))
    )


def main():
    add_metadata()
    # add_metadata_generated()


if __name__ == "__main__":
    main()
