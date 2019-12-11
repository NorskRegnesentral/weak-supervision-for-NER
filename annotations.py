
import numpy as np
import uuid, re, subprocess, os, shutil, math, re, logging, pickle, json, itertools, functools, psutil, time
import multiprocessing, gc
import pandas
import spacy_wrapper, spacy
import snips_nlu_parsers
import utils

# Data files for gazetteers
WIKIDATA = "./data/wikidata.json"
WIKIDATA_SMALL = "./data/wikidata_small.json"
#COMPANY_NAMES = "./data/company_names.json"
GEONAMES = "./data/geonames.json"
CRUNCHBASE = "./data/crunchbase.json"
PRODUCTS = "./data/products.json"
FIRST_NAMES = "./data/first_names.json"


# sets of tokens used for the shallow patterns
MONTHS = {"January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"}
MONTHS_ABBRV =  {"Jan.", "Feb.", "Mar.", "Apr.", "May.", "Jun.", "Jul.", "Aug.", "Sep.", "Sept.", "Oct.", "Nov.", "Dec."}
DAYS = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
DAYS_ABBRV = {"Mon.", "Tu.", "Tue.", "Tues.", "Wed.", "Th.", "Thu.", "Thur.", "Thurs.", "Fri.", "Sat.", "Sun."}
MAGNITUDES = {"million", "billion", "mln", "bln", "bn", "thousand", "m", "k", "b", "m.", "k.", "b.", "mln.", "bln.", "bn."}     
UNITS = {"tons", "tonnes", "barrels", "m", "km", "miles", "kph", "mph", "kg", "°C", "dB", "ft", "gal", "gallons", "g", "kW", "s", "oz",
        "m2", "km2", "yards", "W", "kW", "kWh", "kWh/yr", "Gb", "MW", "kilometers", "meters", "liters", "litres", "g", "grams", "tons/yr",
        'pounds', 'cubits', 'degrees', 'ton', 'kilograms', 'inches', 'inch', 'megawatts', 'metres', 'feet', 'ounces', 'watts', 'megabytes',
        'gigabytes', 'terabytes', 'hectares', 'centimeters', 'millimeters'}
ORDINALS = ({"first, second, third", "fourth", "fifth", "sixth", "seventh"} | 
            {"%i1st"%i for i in range(100)} | {"%i2nd"%i for i in range(100)} | {"%ith"%i for i in range(1000)})
ROMAN_NUMERALS = {'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX','X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII',
                'XVIII', 'XIX', 'XX', 'XXI', 'XXII', 'XXIII', 'XXIV', 'XXV', 'XXVI', 'XXVII', 'XXVIII', 'XXIX', 'XXX'}

# Full list of country names
COUNTRIES = {'Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Antigua', 'Argentina', 'Armenia', 'Australia', 'Austria', 
             'Azerbaijan', 'Bahamas', 'Bahrain',  'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bhutan', 
             'Bolivia', 'Bosnia Herzegovina', 'Botswana', 'Brazil', 'Brunei', 'Bulgaria',  'Burkina', 'Burundi', 'Cambodia', 'Cameroon', 
             'Canada', 'Cape Verde', 'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia', 'Comoros', 'Congo', 'Costa Rica', 
             'Croatia', 'Cuba', 'Cyprus', 'Czech Republic', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'East Timor', 
             'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Ethiopia', 'Fiji', 'Finland', 'France', 
             'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece',  'Grenada', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 
             'Haiti', 'Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Ivory Coast',
             'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati', 'Korea North', 'Korea South', 'Kosovo', 'Kuwait', 'Kyrgyzstan',
             'Laos','Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Macedonia', 'Madagascar', 
             'Malawi', 'Malaysia', 'Maldives','Mali', 'Malta', 'Marshall Islands', 'Mauritania', 'Mauritius', 'Mexico', 'Micronesia', 
             'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique','Myanmar', 'Namibia', 'Nauru', 'Nepal', 'Netherlands', 
             'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'Norway', 'Oman', 'Pakistan', 'Palau', 'Panama', 'Papua New Guinea',
             'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Romania', 'Russian Federation', 'Rwanda', 'St Kitts & Nevis', 
             'St Lucia', 'Saint Vincent & the Grenadines','Samoa', 'San Marino', 'Sao Tome & Principe', 'Saudi Arabia', 'Senegal', 'Serbia', 
             'Seychelles', 'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa', 'South Sudan', 
             'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Swaziland', 'Sweden', 'Switzerland', 'Syria', 'Taiwan', 'Tajikistan', 'Tanzania', 
             'Thailand', 'Togo', 'Tonga', 'Trinidad & Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Tuvalu', 'Uganda', 'Ukraine', 
             'United Arab Emirates', 'United Kingdom', 'United States', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Vatican City', 'Venezuela', 
             'Vietnam', 'Yemen', 'Zambia', 'Zimbabwe', "USA", "UK", "Russia", "South Korea"}

# Natialities, religious and political groups
NORPS = {'Afghan', 'African', 'Albanian', 'Algerian', 'American', 'Andorran', 'Anglican', 'Angolan', 'Arab',  'Aramean','Argentine', 'Armenian', 
         'Asian', 'Australian', 'Austrian', 'Azerbaijani', 'Bahamian', 'Bahraini', 'Baklan', 'Bangladeshi', 'Batswana', 'Belarusian', 'Belgian',
         'Belizean', 'Beninese', 'Bermudian', 'Bhutanese', 'Bolivian', 'Bosnian', 'Brazilian', 'British', 'Bruneian', 'Buddhist', 
         'Bulgarian', 'Burkinabe', 'Burmese', 'Burundian', 'Californian', 'Cambodian', 'Cameroonian', 'Canadian', 'Cape Verdian', 'Catholic', 'Caymanian', 
         'Central African',  'Central American', 'Chadian', 'Chilean', 'Chinese', 'Christian', 'Christian-Democrat', 'Christian-Democratic', 
         'Colombian', 'Communist', 'Comoran', 'Congolese', 'Conservative', 'Costa Rican', 'Croat', 'Cuban', 'Cypriot', 'Czech', 'Dane',  'Danish', 
         'Democrat', 'Democratic', 'Djibouti', 'Dominican', 'Dutch', 'East European', 'Ecuadorean', 'Egyptian', 'Emirati', 'English', 'Equatoguinean',
         'Equatorial Guinean', 'Eritrean', 'Estonian', 'Ethiopian', 'Eurasian', 'European', 'Fijian', 'Filipino', 'Finn', 'Finnish', 'French', 
         'Gabonese', 'Gambian', 'Georgian', 'German', 'Germanic', 'Ghanaian', 'Greek', 'Greenlander', 'Grenadan', 'Grenadian', 'Guadeloupean', 'Guatemalan',
         'Guinea-Bissauan', 'Guinean', 'Guyanese', 'Haitian', 'Hawaiian', 'Hindu', 'Hinduist', 'Hispanic', 'Honduran', 'Hungarian', 'Icelander', 'Indian', 
         'Indonesian', 'Iranian', 'Iraqi', 'Irish', 'Islamic','Islamist', 'Israeli', 'Israelite', 'Italian', 'Ivorian', 'Jain', 'Jamaican', 'Japanese', 
         'Jew',  'Jewish', 'Jordanian', 'Kazakhstani', 'Kenyan', 'Kirghiz', 'Korean', 'Kurd', 'Kurdish',  'Kuwaiti', 'Kyrgyz', 'Labour', 'Latin',
         'Latin American', 'Latvian', 'Lebanese', 'Liberal',  'Liberian', 'Libyan', 'Liechtensteiner', 'Lithuanian', 'Londoner', 'Luxembourger', 
         'Macedonian', 'Malagasy', 'Malawian','Malaysian', 'Maldivan', 'Malian', 'Maltese', 'Manxman', 'Marshallese', 'Martinican', 'Martiniquais', 
         'Marxist', 'Mauritanian', 'Mauritian', 'Mexican', 'Micronesian', 'Moldovan', 'Mongolian', 'Montenegrin', 'Montserratian', 'Moroccan', 
         'Motswana', 'Mozambican', 'Muslim', 'Myanmarese', 'Namibian',  'Nationalist', 'Nazi', 'Nauruan', 'Nepalese', 'Netherlander', 'New Yorker',
         'New Zealander', 'Nicaraguan', 'Nigerian', 'Nordic', 'North American', 'North Korean','Norwegian','Orthodox', 'Pakistani', 'Palauan', 
         'Palestinian', 'Panamanian', 'Papua New Guinean', 'Paraguayan', 'Parisian', 'Peruvian', 'Philistine', 'Pole', 'Polish', 'Portuguese', 
         'Protestant', 'Puerto Rican', 'Qatari', 'Republican', 'Roman', 'Romanian', 'Russian', 'Rwandan', 'Saint Helenian', 'Saint Lucian',   
         'Saint Vincentian', 'Salvadoran', 'Sammarinese', 'Samoan', 'San Marinese', 'Sao Tomean', 'Saudi', 'Saudi Arabian', 'Scandinavian', 'Scottish', 
         'Senegalese', 'Serb', 'Serbian', 'Shia', 'Shiite', 'Sierra Leonean', 'Sikh', 'Singaporean', 'Slovak', 'Slovene', 'Social-Democrat', 'Socialist', 
         'Somali', 'South African', 'South American', 'South Korean', 'Soviet', 'Spaniard', 'Spanish', 'Sri Lankan', 'Sudanese', 'Sunni', 
         'Surinamer', 'Swazi', 'Swede', 'Swedish', 'Swiss', 'Syrian', 'Taiwanese', 'Tajik', 'Tanzanian', 'Taoist', 'Texan', 'Thai', 'Tibetan', 
         'Tobagonian', 'Togolese', 'Tongan', 'Tunisian', 'Turk', 'Turkish', 'Turkmen(s)', 'Tuvaluan', 'Ugandan', 'Ukrainian', 'Uruguayan', 'Uzbek', 
         'Uzbekistani', 'Venezuelan', 'Vietnamese', 'Vincentian', 'Virgin Islander', 'Welsh', 'West European', 'Western', 'Yemeni', 'Yemenite', 
         'Yugoslav', 'Zambian', 'Zimbabwean', 'Zionist'}
                
# Facilities
FACILITIES = {"Palace", "Temple", "Gate", "Museum", "Bridge", "Road", "Airport", "Hospital", "School", "Tower", "Station", "Avenue", 
             "Prison", "Building", "Plant", "Shopping Center", "Shopping Centre", "Mall", "Church", "Synagogue", "Mosque", "Harbor", "Harbour", 
              "Rail", "Railway", "Metro", "Tram", "Highway", "Tunnel", 'House', 'Field', 'Hall', 'Place', 'Freeway', 'Wall', 'Square', 'Park', 
              'Hotel'}

# Legal documents
LEGAL = {"Law", "Agreement", "Act", 'Bill', "Constitution", "Directive", "Treaty", "Code", "Reform", "Convention", "Resolution", "Regulation", 
         "Amendment", "Customs", "Protocol", "Charter"}

# event names
EVENTS = {"War", "Festival", "Show", "Massacre", "Battle", "Revolution", "Olympics", "Games", "Cup", "Week", "Day", "Year", "Series"}

# Names of languages
LANGUAGES = {'Afar', 'Abkhazian', 'Avestan', 'Afrikaans', 'Akan', 'Amharic', 'Aragonese', 'Arabic', 'Aramaic', 'Assamese', 'Avaric', 'Aymara', 
             'Azerbaijani', 'Bashkir',  'Belarusian', 'Bulgarian', 'Bambara', 'Bislama', 'Bengali', 'Tibetan', 'Breton', 'Bosnian', 'Cantonese', 
             'Catalan', 'Chechen',  'Chamorro', 'Corsican', 'Cree', 'Czech', 'Chuvash', 'Welsh',  'Danish', 'German', 'Divehi', 'Dzongkha', 'Ewe', 
             'Greek', 'English', 'Esperanto', 'Spanish', 'Castilian',  'Estonian', 'Basque', 'Persian', 'Fulah', 'Filipino', 'Finnish', 'Fijian', 'Faroese', 
             'French', 'Western Frisian', 'Irish', 'Gaelic', 'Galician', 'Guarani', 'Gujarati', 'Manx', 'Hausa', 'Hebrew', 'Hindi', 'Hiri Motu', 
             'Croatian', 'Haitian', 'Hungarian', 'Armenian', 'Herero', 'Indonesian', 'Igbo', 'Inupiaq', 'Ido', 'Icelandic', 'Italian', 'Inuktitut', 
             'Japanese', 'Javanese', 'Georgian', 'Kongo', 'Kikuyu', 'Kuanyama', 'Kazakh', 'Kalaallisut', 'Greenlandic', 'Central Khmer', 'Kannada', 
             'Korean', 'Kanuri', 'Kashmiri', 'Kurdish','Komi', 'Cornish', 'Kirghiz', 'Latin', 'Luxembourgish', 'Ganda', 'Limburgish', 'Lingala', 'Lao', 
             'Lithuanian', 'Luba-Katanga', 'Latvian', 'Malagasy', 'Marshallese', 'Maori', 'Macedonian', 'Malayalam', 'Mongolian', 'Marathi', 'Malay', 
             'Maltese', 'Burmese', 'Nauru', 'Bokmål', 'Norwegian', 'Ndebele', 'Nepali', 'Ndonga', 'Dutch', 'Flemish', 'Nynorsk', 'Navajo', 'Chichewa', 
             'Occitan', 'Ojibwa', 'Oromo', 'Oriya', 'Ossetian', 'Punjabi', 'Pali', 'Polish', 'Pashto', 'Portuguese', 'Quechua', 'Romansh', 'Rundi', 
             'Romanian', 'Russian', 'Kinyarwanda', 'Sanskrit', 'Sardinian', 'Sindhi', 'Sami', 'Sango', 'Sinhalese',  'Slovak', 'Slovenian', 'Samoan', 
             'Shona', 'Somali', 'Albanian', 'Serbian', 'Swati', 'Sotho', 'Sundanese', 'Swedish', 'Swahili', 'Tamil', 'Telugu', 'Tajik', 'Thai', 
             'Tigrinya', 'Turkmen', 'Taiwanese', 'Tagalog', 'Tswana', 'Tonga', 'Turkish', 'Tsonga', 'Tatar', 'Twi', 'Tahitian', 'Uighur', 'Ukrainian', 
             'Urdu', 'Uzbek', 'Venda', 'Vietnamese', 'Volapük', 'Walloon', 'Wolof', 'Xhosa', 'Yiddish', 'Yoruba', 'Zhuang', 'Mandarin', 
             'Mandarin Chinese',  'Chinese', 'Zulu'}


# Generic words that may appear in official company names but are sometimes skipped when mentioned in news articles (e.g. Nordea Bank -> Nordea)
GENERIC_TOKENS = {"International", "Group", "Solutions", "Technologies", "Management", "Association", "Associates", "Partners", 
                  "Systems", "Holdings", "Services", "Bank", "Fund",  "Stiftung", "Company"}

# List of tokens that are typically lowercase even when they occur in capitalised segments (e.g. International Council of Shopping Centers)
LOWERCASED_TOKENS = {"'s", "-", "a", "an", "the", "at", "by", "for", "in", "of", "on", "to", "up", "and"}

# Prefixes to family names that are often in lowercase
NAME_PREFIXES = {"-", "von", "van", "de", "di", "le", "la", "het", "'t'", "dem", "der", "den", "d'", "ter"}


############################################
# ANNOTATOR
############################################


class BaseAnnotator:
    """Base class for the annotations.  """
    
    def __init__(self, to_exclude=None):
        self.to_exclude = list(to_exclude) if to_exclude is not None else []
        
    def pipe(self, docs):
        """Goes through the stream of documents and annotate them"""
        
        for doc in docs:
            yield self.annotate(doc)     
                     
    def annotate(self, doc):
        """Annotates one single document"""
        
        raise NotImplementedError()
    
    
    def clear_source(self, doc, source):
        """Clears the annotation associated with a given source name"""
        
        if "annotations" not in doc.user_data:
            doc.user_data["annotations"] = {}
        doc.user_data["annotations"][source] = {}
    
    
    def add(self, doc, start, end, label, source, conf=1.0):
        """ Adds a labelled span to the annotation"""
        
        if not self._is_allowed_span(doc, start, end):
            return
        elif (start,end) not in doc.user_data["annotations"][source]:  
            doc.user_data["annotations"][source][(start, end)] = ((label, conf),)
        
        # If the span is already present, we need to check that the total confidence does not exceed 1.0
        else:  
            current_vals = doc.user_data["annotations"][source][(start, end)]
            if label in {label2 for label2, _ in current_vals}:
                return
            total_conf = sum([conf2 for _, conf2 in current_vals]) + conf
            if total_conf > 1.0:
                current_vals = [(label2, conf2/total_conf) for label2, conf2 in current_vals]
                conf = conf/total_conf
            doc.user_data["annotations"][source][(start, end)] = (*current_vals, (label, conf))
                
        
        
    def _is_allowed_span(self, doc, start, end):
        """Checks whether the span is allowed (given exclusivity relations with other sources)"""
        for source in self.to_exclude:
            intervals = list(doc.user_data["annotations"][source].keys())
           
            start_search, end_search = _binary_search(start, end, intervals)        
            for interval_start, interval_end in intervals[start_search:end_search]:
                if start < interval_end and end > interval_start:
                    return False

        return True
        

    def annotate_docbin(self, docbin_input_file, docbin_output_file=None, return_raw=False,
                        cutoff=None, nb_to_skip=0):
        """Runs the annotator on the documents of a DocBin file, and write the output
        to the same file (or returns the raw data is return_raw is True)"""

        attrs = [spacy.attrs.LEMMA, spacy.attrs.TAG, spacy.attrs.DEP, spacy.attrs.HEAD, 
                 spacy.attrs.ENT_IOB, spacy.attrs.ENT_TYPE]
        docbin = spacy.tokens.DocBin(attrs=attrs, store_user_data=True)

        print("Reading", docbin_input_file, end="...", flush=True)
        for doc in self.pipe(docbin_reader(docbin_input_file, cutoff=cutoff, nb_to_skip=nb_to_skip)):
            docbin.add(doc)
            if len(docbin)%1000 ==0:
                print("Number of processed documents:", len(docbin))

        print("Finished annotating", docbin_input_file)
        
        data = docbin.to_bytes()  
        if return_raw:
            return data
        else:
            if docbin_output_file is None:
                docbin_output_file = docbin_input_file
            print("Write to", docbin_output_file, end="...", flush=True)
            fd = open(docbin_output_file, "wb")
            fd.write(data)
            fd.close()
            print("done")

            
class FullAnnotator(BaseAnnotator):
    """Annotator of entities in documents, combining several sub-annotators (such as gazetteers, 
    spacy models etc.). To add all annotators currently implemented, call add_all(). """
      
    def __init__(self):
        super(FullAnnotator,self).__init__()
        self.annotators = []
    
    def pipe(self, docs):
        """Annotates the stream of documents using the sub-annotators."""
        
        streams = itertools.tee(docs, len(self.annotators)+1)
        
        pipes = [annotator.pipe(stream) for annotator,stream in zip(self.annotators, streams[1:])]
        
        for doc in streams[0]:
            for pipe in pipes:
                try:
                    next(pipe)
                except:
                    print("ignoring document:", doc)
            yield doc
            
    def annotate(self, doc):
        """Annotates a single  document with the sub-annotators
        NB: do not use this method for large collections of documents (as it is quite inefficient), and
        prefer the method pipe that runs the Spacy models on batches of documents"""
        
        for annotator in self.annotators:
            doc = annotator.annotate(doc)
        return doc


    def add_annotator(self, annotator):
        self.annotators.append(annotator)
        return self
            
          
    def add_all(self):
        """Adds all implemented annotation functions, models and filters"""
        
        print("Loading shallow functions")
        self.add_shallow()
        print("Loading Spacy NER models")
        self.add_models()
        print("Loading gazetteer supervision modules")
        self.add_gazetteers()        
        print("Loading document-level supervision sources")
        self.add_doc_level()
                      
        return self
    
    def add_shallow(self):
        """Adds shallow annotation functions"""
        
        # Detection of dates, time, money, and numbers
        self.add_annotator(FunctionAnnotator(date_generator, "date_detector"))
        self.add_annotator(FunctionAnnotator(time_generator, "time_detector"))
        self.add_annotator(FunctionAnnotator(money_generator, "money_detector"))
        exclusives = ["date_detector", "time_detector", "money_detector"]
        
        # Detection based on casing
        proper_detector = SpanGenerator(is_likely_proper)
        self.add_annotator(FunctionAnnotator(proper_detector, "proper_detector", 
                                             to_exclude=exclusives))

        # Detection based on casing, but allowing some lowercased tokens
        proper2_detector = SpanGenerator(is_likely_proper, exceptions=LOWERCASED_TOKENS)
        self.add_annotator(FunctionAnnotator(proper2_detector, "proper2_detector", 
                                             to_exclude=exclusives))
        
        # Detection based on part-of-speech tags
        nnp_detector = SpanGenerator(lambda tok: tok.tag_=="NNP")
        self.add_annotator(FunctionAnnotator(nnp_detector, "nnp_detector", 
                                             to_exclude=exclusives))
        
        # Detection based on dependency relations (compound phrases)
        compound_detector = SpanGenerator(lambda x: is_likely_proper(x) and in_compound(x))
        self.add_annotator(FunctionAnnotator(compound_detector, "compound_detector", 
                                             to_exclude=exclusives))
                            
        # We add one variants for each NE detector, looking at infrequent tokens
        for source_name in ["proper_detector", "proper2_detector", "nnp_detector","compound_detector"]:
            self.add_annotator(SpanConstraintAnnotator(is_infrequent, source_name, "infrequent_"))

        self.add_annotator(FunctionAnnotator(legal_generator, "legal_detector", exclusives))
        exclusives += ["legal_detector"]
        self.add_annotator(FunctionAnnotator(number_generator, "number_detector", exclusives))

        # Detection of companies with a legal type
        self.add_annotator(FunctionAnnotator(CompanyTypeGenerator(), "company_type_detector", 
                                             to_exclude=exclusives))
        
        # Detection of full person names
        self.add_annotator(FunctionAnnotator(FullNameGenerator(), "full_name_detector", 
                                             to_exclude=exclusives+["company_type_detector"]))

        # Detection based on a probabilistic parser
        self.add_annotator(FunctionAnnotator(SnipsGenerator(), "snips"))
                
        return self
    
        
    def add_models(self):
        """Adds Spacy NER models to the annotator"""
        
        self.add_annotator(ModelAnnotator("en_core_web_md", "core_web_md"))
        self.add_annotator(ModelAnnotator("data/conll2003", "conll2003"))
        self.add_annotator(ModelAnnotator("data/BTC", "BTC"))
        self.add_annotator(ModelAnnotator("data/SEC-filings", "SEC"))
        
        return self
        
    def add_gazetteers(self):
        """Adds gazetteer supervision models (company names and wikidata)."""
                    
        exclusives = ["date_detector", "time_detector", "money_detector", "number_detector"]
        
         # Annotation of company, person and location names based on wikidata
        self.add_annotator(GazetteerAnnotator(WIKIDATA, "wiki", to_exclude=exclusives))

         # Annotation of company, person and location names based on wikidata (only entries with descriptions)
        self.add_annotator(GazetteerAnnotator(WIKIDATA_SMALL, "wiki_small", to_exclude=exclusives))

         # Annotation of location names based on geonames
        self.add_annotator(GazetteerAnnotator(GEONAMES, "geo", to_exclude=exclusives))
        
         # Annotation of organisation and person names based on crunchbase open data
        self.add_annotator(GazetteerAnnotator(CRUNCHBASE, "crunchbase", to_exclude=exclusives))
        
        # Annotation of product names
        self.add_annotator(GazetteerAnnotator(PRODUCTS, "product", to_exclude=exclusives[:-1]))
        
        # We also add new sources for multitoken entities (which have higher confidence)
        for source_name in ["wiki", "wiki_small", "geo", "crunchbase", "product"]:
            for cased in ["cased", "uncased"]:
                self.add_annotator(SpanConstraintAnnotator(lambda s: len(s) > 1, "%s_%s"%(source_name, cased), "multitoken_"))
                
        self.add_annotator(FunctionAnnotator(misc_generator, "misc_detector", exclusives))
        
        return self
            
    
    
    def add_doc_level(self):
        """Adds document-level supervision sources"""
        
        self.add_annotator(StandardiseAnnotator())
        self.add_annotator(DocumentHistoryAnnotator())
        self.add_annotator(DocumentMajorityAnnotator())
        return self
    
 

############################################
# I/O FUNCTIONS
############################################
   
    
def docbin_reader(docbin_input_file, vocab=None, cutoff=None, nb_to_skip=0):
    """Generate Spacy documents from the input file"""
    
    if vocab is None:
        if not hasattr(docbin_reader, "vocab"):
            print("Loading vocabulary", end="...", flush=True)
            docbin_reader.vocab = spacy.load("en_core_web_md").vocab
            print("done")
        vocab = docbin_reader.vocab
    vocab.strings.add("subtok")
        
    fd = open(docbin_input_file, "rb")
    data = fd.read()
    fd.close()
    docbin = spacy.tokens.DocBin(store_user_data=True)
    docbin.from_bytes(data)
    del data
   # print("Total number of documents in docbin:", len(docbin))

    # Hack to easily skip a number of documents
    if nb_to_skip:
        docbin.tokens = docbin.tokens[nb_to_skip:]
        docbin.spaces = docbin.spaces[nb_to_skip:]
        docbin.user_data = docbin.user_data[nb_to_skip:]
        
    reader = docbin.get_docs(vocab) 
    for i, doc in enumerate(reader):
        yield doc
        if cutoff is not None and (i+1) >= cutoff:
            return
    


                   
def convert_to_json(docbin_file, json_file, source="HMM", cutoff=None, nb_to_skip=0):
     
    corpus_gen = docbin_reader(docbin_file)
    for i in range(nb_to_skip):
        next(corpus_gen)
        
    print("Writing JSON file to", json_file)
    out_fd = open(json_file, "wt")
    out_fd.write("[{\"id\": 0, \"paragraphs\": [\n")
    for i, doc in enumerate(corpus_gen):
        doc.ents = tuple(spacy.tokens.Span(doc, start, end, doc.vocab.strings[label]) 
                          for (start, end), ((label,conf),) in doc.user_data["annotations"][source].items() if end <= len(doc))
        doc.user_data = {}
        d = spacy.gold.docs_to_json([doc])
        s = json.dumps(d["paragraphs"]).strip("[]")
        if i==0:
            out_fd.write(s)
        else:
            out_fd.write(",\n"+s)
        if cutoff is not None and i >= cutoff:
            out_fd.flush()
            break
        if i>0 and i % 1000 == 0:
            print("Converted documents:", i)
            out_fd.flush() 
    out_fd.write("]}]\n")
    out_fd.flush()
    out_fd.close()
    
        

def _annotate_docbin_parallel(cmd, docbin_input_file, docbin_output_file, nb_cores=8, nb_docs=160000):
    """Utility function to easily distribute the annotation of large docbin files on 
    several cores. The command must be a string whose evaluation returns the annotator"""
    p = multiprocessing.Pool(nb_cores)
    args = [(cmd, docbin_input_file, (nb_docs//nb_cores) if i<nb_cores-1 else None, 
             i*(nb_docs//nb_cores)) for i in range(nb_cores)]
    datas = p.starmap(_annotate_fun, args)
    p.close()
    attrs = [spacy.attrs.LEMMA, spacy.attrs.TAG, spacy.attrs.DEP, spacy.attrs.HEAD, 
                 spacy.attrs.ENT_IOB, spacy.attrs.ENT_TYPE]
    docbin = spacy.tokens.DocBin(attrs, store_user_data=True)
    for data in datas:
        docbin.merge(spacy.tokens.DocBin(store_user_data=True).from_bytes(data))
        
    data = docbin.to_bytes()            
    print("Write to", docbin_output_file, end="...", flush=True)
    fd = open(docbin_output_file, "wb")
    fd.write(data)
    fd.close()
    print("done")
        
def _annotate_fun(cmd, docbin_input_file, cutoff, nb_to_skip):
    return eval(cmd).annotate_docbin(docbin_input_file, cutoff=cutoff, nb_to_skip=nb_to_skip)


############################################
# CORE ANNOTATORS
############################################

      
 
class ModelAnnotator(BaseAnnotator):
    """Annotation based on a spacy NER model"""
    
    def __init__(self, model_path, source_name):
        super(ModelAnnotator, self).__init__()
        print("loading", model_path, end="...", flush=True)
        model = spacy.load(model_path)
        self.ner = model.get_pipe("ner")
        self.source_name = source_name
        print("done")
        
        
    def pipe(self, docs): 
        """Annotates the stream of documents based on the Spacy NER model"""    
            
        stream1, stream2 = itertools.tee(docs, 2)
        
        # Apply the NER models through the pipe
        # (we need to work on copies to strange deadlock conditions)
        stream2 = (spacy.tokens.Doc(d.vocab).from_bytes(d.to_bytes(exclude="user_data"))
                 for d in stream2)
        def remove_ents(doc): doc.ents = tuple() ; return doc
        stream2 = (remove_ents(d) for d in stream2)
        stream2 = self.ner.pipe(stream2)
             
        for doc, doc_copy in zip(stream1, stream2):
            
            self.clear_source(doc, self.source_name)
            self.clear_source(doc, self.source_name+"+c")
            
            # Add the annotation
            for ent in doc_copy.ents:
                self.add(doc, ent.start, ent.end, ent.label_, self.source_name)
            
            # Correct some entities
            doc_copy = spacy_wrapper._correct_entities(doc_copy) 
            for ent in doc_copy.ents:
                self.add(doc, ent.start, ent.end, ent.label_, self.source_name+"+c")
            
            yield doc
                
                
    def annotate(self, doc):
        """Annotates one single document using the Spacy NER model
        NB: do not use this method for large collections of documents (as it is quite inefficient), and
        prefer the method pipe that runs the Spacy model on batches of documents"""
        
        ents = list(doc.ents)
        doc.ents = tuple()
        doc = self.ner(doc)
        
        self.clear_source(doc, self.source_name)
        self.clear_source(doc, self.source_name+"+c")
            
        # Add the annotation
        for ent in doc.ents:
            self.add(doc, ent.start, ent.end, ent.label_, self.source_name)
            
        # Correct some entities
        doc = spacy_wrapper._correct_entities(doc) 
        for ent in doc.ents:
            self.add(doc, ent.start, ent.end, ent.label_, self.source_name+"+c")
        
        doc.ents = ents
        return doc
     
  
class FunctionAnnotator(BaseAnnotator):
    """Annotation based on a heuristic function that generates (start,end,label) given a spacy document"""
    
    def __init__(self, function, source_name, to_exclude=()):
        """Create an annotator based on a function generating labelled spans given a Spacy Doc object. Spans that
        overlap with existing spans from sources listed in 'to_exclude' are ignored. """

        super(FunctionAnnotator, self).__init__(to_exclude=to_exclude)

        self.function = function
        self.source_name = source_name
         
            
    def annotate(self, doc):
        """Annotates one single document"""
        
        self.clear_source(doc, self.source_name)
        
        for start, end, label in self.function(doc):
            self.add(doc, start, end, label, self.source_name)               
        return doc
        
                        
                
class SpanConstraintAnnotator(BaseAnnotator):
    """Annotation by looking at text spans (from another source) that satisfy a span-level constratint"""
    
    def __init__(self, constraint, initial_source_name, prefix):
        
        super(SpanConstraintAnnotator, self).__init__()
        
        self.constraint = constraint
        self.initial_source_name = initial_source_name
        self.prefix = prefix

            
    def annotate(self, doc):
        """Annotates one single document"""
        
        self.clear_source(doc, self.prefix+self.initial_source_name)
        
        for (start, end), vals in doc.user_data["annotations"][self.initial_source_name].items():
            if self.constraint(doc[start:end]):
                for label, conf in vals:
                    self.add(doc, start, end, label, self.prefix+self.initial_source_name, conf)               
          
        return doc
    
 
############################################
# ANNOTATION WITH DISTANT SUPERVISION
############################################
 

class GazetteerAnnotator(BaseAnnotator):
    """Annotation using a gazetteer, i.e. a large list of entity terms. The annotation looks 
    both at case-sensitive and case-insensitive occurrences.  The annotator relies on a token-level 
    trie for efficient search. """
    
    def __init__(self, json_file, source_name, to_exclude=()):
        
        super(GazetteerAnnotator, self).__init__(to_exclude=to_exclude)

        self.trie = extract_json_data(json_file)
        self.source_name = source_name
         
            
    def annotate(self, doc):
        """Annotates one single document"""
        
        self.clear_source(doc, "%s_%s"%(self.source_name, "cased"))
        self.clear_source(doc, "%s_%s"%(self.source_name, "uncased"))
        
        for start, end, label, conf in self.get_hits(doc, case_sensitive=True, full_compound=True):
            
            self.add(doc, start, end, label, "%s_%s"%(self.source_name, "cased"), conf)

        for start, end, label, conf in self.get_hits(doc, case_sensitive=False, full_compound=True):
            
            self.add(doc, start, end, label, "%s_%s"%(self.source_name, "uncased"), conf)
                    
        return doc
                    
                        
    def get_hits(self, spacy_doc, case_sensitive=True, lookahead=10, full_compound=True):
        """Search for occurrences of entity terms in the spacy document"""
        
        tokens = tuple(tok.text for tok in spacy_doc)
        
        i = 0
        while i < len(tokens):
            
            tok = spacy_doc[i]
            # Skip punctuation
            if tok.is_punct:
                i += 1
                continue
            
            # We skip if we are inside a compound phrase
            elif full_compound and i > 0 and is_likely_proper(spacy_doc[i-1]) and spacy_doc[i-1].dep_=="compound":
                i += 1
                continue
    
            span = tokens[i:i+lookahead]
            prefix_length, prefix_value = self.trie.longest_prefix(span, case_sensitive)
            if prefix_length:
                
                # We further require at least one proper noun token (to avoid too many FPs)
                if not any(is_likely_proper(tok) for tok in spacy_doc[i:i+prefix_length]):
                    i += 1
                    continue
                
                # If we found a company and the next token is a legal suffix, include it
                if ((i+prefix_length) < len(spacy_doc) and {"ORG", "COMPANY"}.intersection(prefix_value) 
                    and spacy_doc[i+prefix_length].lower_.rstrip(".") in spacy_wrapper.LEGAL_SUFFIXES):
                    prefix_length += 1
                    
                # If the following tokens are part of the same compound phrase, skip
                if full_compound and spacy_doc[i+prefix_length-1].dep_=="compound" and spacy_doc[i+prefix_length].text not in {"'s"}:
                    i += 1
                    continue
                    
                # Must account for spans with multiple possible entities
                for neClass in prefix_value:
                    yield i, i+prefix_length, neClass, 1/len(prefix_value)
                    
                # We skip the text until the end of the occurences + 1 token (we assume two entities do not
                # follow one another without at least one token such as a comma)
                i += (prefix_length + 1)
            else:
                i += 1       
                
    
def extract_json_data(json_file):
    """Extract entities from a Json file """
    
    print("Extracting data from", json_file)
    fd = open(json_file)
    data = json.load(fd)
    fd.close()
    trie = utils.Trie()
    for neClass, names in data.items():
        print("Populating trie for entity class %s (number: %i)"%(neClass, len(names)))
        for name in names:
            
            # Removes parentheses and appositions
            name = name.split("(")[0].split(",")[0].rstrip()
            
            name = tuple(utils.tokenise_fast(name))
            # Add the entity into the trie (values are tuples since each entity may have several possible types) 
            if name in trie and neClass not in trie[name]:
                trie[name] = (*trie[name], neClass)
            else:
                trie[name] = (neClass,)
    return trie
        

            
############################################
# ANNOTATION WITH SHALLOW PATTERNS
############################################

            
def date_generator(spacy_doc):
    """Searches for occurrences of date patterns in text"""
    
    spans = {}
    
    i = 0
    while i < len(spacy_doc):
        tok = spacy_doc[i]
        if tok.lemma_ in DAYS | DAYS_ABBRV:
            spans[(i,i+1)] = "DATE"
        elif tok.is_digit and re.match("\d+$", tok.text) and int(tok.text) > 1920 and int(tok.text) < 2040:
            spans[(i,i+1)] = "DATE"
        elif tok.lemma_ in MONTHS | MONTHS_ABBRV:       
            if tok.tag_=="MD": # Skipping "May" used as auxiliary
                pass
            elif i > 0 and re.match("\d+$", spacy_doc[i-1].text) and int(spacy_doc[i-1].text) < 32:
                spans[(i-1,i+1)] = "DATE"
            elif i > 1 and re.match("\d+(?:st|nd|rd|th)$", spacy_doc[i-2].text) and spacy_doc[i-1].lower_=="of":
                spans[(i-2,i+1)] = "DATE"                
            elif i < len(spacy_doc)-1 and re.match("\d+$", spacy_doc[i+1].text) and int(spacy_doc[i+1].text) < 32: 
                spans[(i,i+2)] = "DATE"
                i += 1
            else:
                spans[(i,i+1)] = "DATE"                    
        i += 1
    
    # Concatenating contiguous spans
    spans = merge_contiguous_spans(spans, spacy_doc)
    
    for i, ((start,end), content) in enumerate(spans.items()):
            yield start, end, content
                
             
    
def time_generator(spacy_doc):
    """Searches for occurrences of time patterns in text"""
    
    i = 0
    while i < len(spacy_doc):
        tok = spacy_doc[i]

        if (i < len(spacy_doc)-1 and tok.text[0].isdigit() and 
              spacy_doc[i+1].lower_ in {"am", "pm", "a.m.", "p.m.", "am.", "pm."}):
            yield i, i+2, "TIME"
            i += 1
        elif tok.text[0].isdigit() and re.match("\d{1,2}\:\d{1,2}", tok.text):
            yield i, i+1, "TIME"
            i += 1
        i += 1
    
    
       
def money_generator(spacy_doc):
    """Searches for occurrences of money patterns in text"""      
        
    i = 0
    while i < len(spacy_doc):
        tok = spacy_doc[i]
        if tok.text[0].isdigit():
            j = i+1
            while (j < len(spacy_doc) and (spacy_doc[j].text[0].isdigit() or  spacy_doc[j].norm_ in MAGNITUDES)):
                j += 1
                
            found_symbol = False
            if i > 0 and spacy_doc[i-1].text in (spacy_wrapper.CURRENCY_CODES | 
                                                 spacy_wrapper.CURRENCY_SYMBOLS):
                i = i-1
                found_symbol = True
            if j < len(spacy_doc) and spacy_doc[j].text in (spacy_wrapper.CURRENCY_CODES | 
                                                            spacy_wrapper.CURRENCY_SYMBOLS | 
                                                            {"euros", "cents", "rubles"}):
                j += 1
                found_symbol = True
                
            if found_symbol:
                yield i,j, "MONEY"
            i = j
        else:
            i += 1
            
    
    
def number_generator(spacy_doc):
    """Searches for occurrences of number patterns (cardinal, ordinal, quantity or percent) in text"""

   
    i = 0
    while i < len(spacy_doc):
        tok = spacy_doc[i]
    
        if tok.lower_ in ORDINALS:
            yield i, i+1, "ORDINAL"
            
        elif re.search("\d", tok.text):
            j = i+1
            while (j < len(spacy_doc) and (spacy_doc[j].norm_ in MAGNITUDES)):
                j += 1
            if j < len(spacy_doc) and spacy_doc[j].lower_.rstrip(".") in UNITS:
                j += 1
                yield i, j, "QUANTITY"
            elif j < len(spacy_doc) and spacy_doc[j].lower_ in ["%", "percent", "pc.", "pc", "pct", "pct.", "percents", "percentage"]:
                j += 1
                yield i, j, "PERCENT"        
            else:
                yield i, j,  "CARDINAL"
            i = j-1
        i += 1
 
    
class SpanGenerator:
    """Generate spans that satisfy a token-level constratint"""
    
    def __init__(self, constraint, label="ENT", exceptions=("'s", "-")):
        """annotation with a constraint (on spacy tokens). Exceptions are sets of tokens that are allowed
        to violate the constraint inside the span"""
        
        self.constraint = constraint
        self.label = label
        self.exceptions = set(exceptions)
        
    def __call__(self, spacy_doc):    

        i = 0
        while i < len(spacy_doc):
            tok = spacy_doc[i]
                # We search for the longest span that satisfy the constraint
            if self.constraint(tok):
                j = i+1
                while True:
                    if j < len(spacy_doc) and self.constraint(spacy_doc[j]):
                        j += 1
                    # We relax the constraint a bit to allow genitive and dashes
                    elif j < (len(spacy_doc)-1) and spacy_doc[j].text in self.exceptions and self.constraint(spacy_doc[j+1]):
                        j += 2
                    else:
                        break

                # To avoid too many FPs, we only keep entities with at least 3 characters (excluding punctuation)
                if len(spacy_doc[i:j].text.rstrip(".")) > 2:
                    yield i, j, self.label
                i = j
            else:
                i += 1
                
                        
    
class CompanyTypeGenerator:
    """Search for compound spans that end with a legal suffix"""
    
    def __init__(self):
        self.suggest_generator = SpanGenerator(lambda x: is_likely_proper(x) and in_compound(x))
        
    def __call__(self, spacy_doc):
        
        for start, end, _ in self.suggest_generator(spacy_doc):
            if spacy_doc[end-1].lower_.rstrip(".") in spacy_wrapper.LEGAL_SUFFIXES:
                yield start, end, "COMPANY"
            elif end < len(spacy_doc) and spacy_doc[end].lower_.rstrip(".") in spacy_wrapper.LEGAL_SUFFIXES:
                yield start, end+1, "COMPANY"

                        
class FullNameGenerator:
    """Search for occurrences of full person names (first name followed by at least one title token)"""

    def __init__(self):
        fd = open(FIRST_NAMES)
        self.first_names = set(json.load(fd))
        fd.close()
        self.suggest_generator = SpanGenerator(lambda x: is_likely_proper(x) and in_compound(x), 
                                               exceptions=NAME_PREFIXES)
        
    def __call__(self, spacy_doc):
        
        for start, end, _ in self.suggest_generator(spacy_doc):   
                    
            # We assume full names are between 2 and 4 tokens
            if (end-start) < 2 or (end-start) > 5:
                continue
                
            elif (spacy_doc[start].text in self.first_names and spacy_doc[end-1].is_alpha 
                  and spacy_doc[end-1].is_title): 
                yield start, end, "PERSON"
                
                
                    

class SnipsGenerator:
    """Annotation using the Snips NLU entity parser. """
    
    def __init__(self):
        """Initialise the annotation tool."""
        
        self.parser = snips_nlu_parsers.BuiltinEntityParser.build(language="en") 
    
    def __call__(self, spacy_doc):
        """Runs the parser on the spacy document, and convert the result to labels."""
        
        text = spacy_doc.text
        
        # The current version of Snips has a bug that makes it crash with some rare Turkish characters, or mentions of "billion years"
        text = text.replace("’", "'").replace("”", "\"").replace("“", "\"").replace("—", "-").encode("iso-8859-15", "ignore").decode("iso-8859-15")
        text = re.sub("(\d+) ([bm]illion(?: (?:\d+|one|two|three|four|five|six|seven|eight|nine|ten))? years?)", "\g<1>.0 \g<2>", text)

        results = self.parser.parse(text)
        for result in results:
            span = spacy_doc.char_span(result["range"]["start"], result["range"]["end"])
            if span is None or span.lower_ in {"now"} or span.text in {"may"}:
                continue
            label = None
            if result["entity_kind"]=="snips/number" and span.lower_ not in {"one", "some", "few", "many", "several"}:
                label = "CARDINAL"
            elif result["entity_kind"]=="snips/ordinal" and span.lower_ not in {"first", "second", "the first", "the second"}:
                label = "ORDINAL"
            elif result["entity_kind"]=="snips/amountOfMoney":
                label = "MONEY"
            elif result["entity_kind"]=="snips/percentage":
                label = "PERCENT"
            elif result["entity_kind"] in {"snips/date", "snips/datePeriod"}:
                label = "DATE"
            elif result["entity_kind"] in {"snips/time", "snips/timePeriod"}:
                label = "TIME"
            
            if label: 
                yield span.start, span.end, label

                
def legal_generator(spacy_doc):
   
    legal_spans = {}
    for (start,end) in get_spans(spacy_doc, ["proper2_detector", "nnp_detector"]):
        if not is_likely_proper(spacy_doc[end-1]):
            continue
                
        span = spacy_doc[start:end].text
        last_token = spacy_doc[end-1].text.title().rstrip("s")
                  
        if last_token in LEGAL:     
            legal_spans[(start,end)] = "LAW"
                     
    
    # Handling legal references such as Article 5
    for i in range(len(spacy_doc)-1):
        if spacy_doc[i].text.rstrip("s") in {"Article", "Paragraph", "Section", "Chapter", "§"}:
            if spacy_doc[i+1].text[0].isdigit() or spacy_doc[i+1].text in ROMAN_NUMERALS:
                start, end = i, i+2
                if (i < len(spacy_doc)-3 and spacy_doc[i+2].text in {"-", "to", "and"} 
                    and (spacy_doc[i+3].text[0].isdigit() or spacy_doc[i+3].text in ROMAN_NUMERALS)):
                    end = i+4
                legal_spans[start, end] = "LAW"

    # Merge contiguous spans of legal references ("Article 5, Paragraph 3")
    legal_spans = merge_contiguous_spans(legal_spans, spacy_doc)
    for start, end in legal_spans:
        yield start, end, "LAW"
        

        
def misc_generator(spacy_doc):
    """Detects occurrences of countries and various less-common entities (NORP, FAC, EVENT, LANG)"""
    
    spans = set(spacy_doc.user_data["annotations"]["proper_detector"].keys())
    spans.update((i,i+1) for i in range(len(spacy_doc)))
    spans = sorted(spans, key=lambda x:x[0])
    
    for (start,end) in spans:

        span = spacy_doc[start:end].text
        span = span.title() if span.isupper() else span
        last_token = spacy_doc[end-1].text

        if span in COUNTRIES:
            yield start, end, "GPE"

        if end <= (start+3) and (span in NORPS or last_token in NORPS or last_token.rstrip("s") in NORPS):
            yield start, end, "NORP"
    
        if span in LANGUAGES and spacy_doc[start].tag_=="NNP":
            yield start, end, "LANGUAGE"
            
        if last_token in FACILITIES and end > start+1:
            yield start, end, "FAC"     

        if last_token in EVENTS and end > start+1:
            yield start, end, "EVENT"     
    
    

############################################
# STANDARDISATION OF ANNOTATIONS
############################################


class StandardiseAnnotator(BaseAnnotator):
    """Annotator taking existing annotations and standardising them (i.e. changing PER to PERSON,
    or changing LOC to GPE if other annotations suggest the entity is a GPE)"""
       

    def annotate(self, doc):
        """Annotates one single document"""     
    
        gpe_sources = ["geo_cased", "geo_uncased", "wiki_cased", "wiki_uncased", "core_web_md+c", "doc_majority_cased"]
        company_sources = ["company_type_detector", "crunchbase_cased", "crunchbase_uncased", "doc_majority_cased", "doc_majority_uncased"]
           
        for source in list(doc.user_data.get("annotations", [])):

            if "unified" in source:
                del doc.user_data["annotations"][source]
                continue
            current_spans = dict(doc.user_data["annotations"][source])
            self.clear_source(doc, source)
            for span, vals in current_spans.items():
                new_vals = []
                for label, conf in vals:
                    if label=="PER":
                        label="PERSON"
                    if label=="LOC" and (source.startswith("conll") 
                                              or source.startswith("BTC") 
                                              or source.startswith("SEC")
                                             or source.startswith("doc_majority")):
                        for gpe_source in gpe_sources:
                            if span in doc.user_data["annotations"].get(gpe_source,[]):
                                for label2, conf2 in doc.user_data["annotations"][gpe_source][span]:
                                    if label2 =="GPE":
                                        label="GPE"

                    if label=="ORG" and (source.startswith("conll")
                                             or source.startswith("BTC")
                                             or source.startswith("SEC")
                                             or source.startswith("core_web_md")
                                             or source.startswith("doc_majority")
                                             or "wiki_" in source):
                        for company_source in company_sources:
                            if span in doc.user_data["annotations"].get(company_source,[]):
                                for label2, conf2 in doc.user_data["annotations"][company_source][span]:
                                    if label2 =="COMPANY":
                                        label="COMPANY"
                    new_vals.append((label, conf))
                for label, conf in new_vals:
                    self.add(doc, span[0], span[1], label, source, conf)
                    

        return doc
    
    
############################################
# DOCUMENT-LEVEL ANNOTATION
############################################    


class DocumentHistoryAnnotator(BaseAnnotator):
    """Annotation based on the document history: 
    1) if a person name has been mentioned in full (at least two consecutive tokens, most often first name followed by 
    last name), then mark future occurrences of the last token (last name) as a PERSON as well. 
    2) if a company name has been mentioned together with a legal type, mark all other occurrences (possibly without 
    the legal type at the end) also as a COMPANY.
    """  
            
    def annotate(self, doc):
        """Annotates one single document"""

        self.clear_source(doc, "doc_history")
       
        trie = utils.Trie()     
         
        # If the doc has fields, we start with the longest ones (i.e. the content) 
        if "fields" in doc.user_data:
            field_lengths = {field:(field_end-field_start) for field, (field_start, field_end) in doc.user_data["fields"].items()}
            sorted_fields = sorted(doc.user_data["fields"].keys(), key=lambda x: field_lengths[x], reverse=True)
            field_boundaries = [doc.user_data["fields"][field_name] for field_name in sorted_fields]
        else:
            field_boundaries = [(0, len(doc))]
        
        for field_start, field_end in field_boundaries:

            sub_doc = doc[field_start:field_end]
            tokens = tuple(tok.text for tok in sub_doc)

            all_spans = [((start,end), val) for source in doc.user_data["annotations"] 
                         for ((start,end), val) in doc.user_data["annotations"][source].items()
                         if source  in ["core_web_md+c", "conll2003+c","full_name_detector", "company_type_detector"] 
                         or source.endswith("cased")]

            all_spans = sorted(all_spans, key=lambda x:x[0][0])

            # We search for occurrences of full person names or company names with legal suffix
            for (start,end), val in all_spans:
                if len(val)==0:
                    continue
                if val[0][0]=="PERSON" and end > (start+1) and end < (start +5):  
                    last_name =tokens[end-1:end] 
                    if last_name not in trie:
                        trie[tokens[start:end]] = (start, "PERSON")
                        trie[tokens[end-1:end]] = (start, "PERSON")
                            
                elif (val[0][0] in {"COMPANY", "ORG"} and end > (start+1) and end < (start +8) and 
                    doc[end-1].lower_.rstrip(".") in spacy_wrapper.LEGAL_SUFFIXES):  
                    company_without_suffix = tokens[start:end-1]
                    if company_without_suffix not in trie:
                        trie[tokens[start:end-1]] = (start, "COMPANY")
                        trie[tokens[start:end]] = (start, "COMPANY")
                                
            i = 0
            while i < len(tokens):
            
                span = tokens[i:i+8]
                prefix_length, prefix_value = trie.longest_prefix(span)
                    
                if prefix_length:   
                    initial_offset, label = prefix_value
                    if i >initial_offset:
                        self.add(doc, i, i+prefix_length, label, "doc_history")
                    i += prefix_length
                else:
                    i += 1      
        return doc

                        

class DocumentMajorityAnnotator(BaseAnnotator):
    """Annotation based on majority label for the same entity string elsewhere in the document. The 
    annotation creates two layers, one for case-sensitive occurrences of the entity string in the document,
    and one for case-insensitive occurrences.
    """
           
    
    def annotate(self, doc):
        """Annotates one single document"""     
        
        self.clear_source(doc, "doc_majority_cased")
        self.clear_source(doc, "doc_majority_uncased")

        entity_counts = self.get_counts(doc)
        
        # And we build a trie to easily search for these entities in the text          
        trie = utils.Trie()
        for entity, label_counts in entity_counts.items():
                      
            # We require at least 2 occurences of the text span in the document
            entity_lower = tuple(t.lower() for t in entity)
            nb_occurrences = 0
            
            tokens_lc = tuple(t.lower_ for t in doc)
            for i in range(len(tokens_lc)-len(entity)):
                if tokens_lc[i:i+len(entity)] == entity_lower:
                    nb_occurrences += 1
            
            # We select the majority label (and give a small preference to rarer/more detailed labels)
            majority_label = max(label_counts, key=lambda x: (label_counts.get(x)*1000 + 
                                                              (1 if x in {"PRODUCT", "COMPANY"} else 0)))
            if nb_occurrences > 1:
                trie[entity] = majority_label

        # Searching for case-sensitive occurrences of the entities
        self.add_annotations(doc, trie)
        self.add_annotations(doc, trie, False)
                  
        return doc

    
    def get_counts(self, doc):

        entity_counts = {}
            
        # We first count the possible labels for each span
        span_labels = {}
        
        sources = ['BTC', 'BTC+c', 'company_type_detector', 'conll2003', 'conll2003+c','core_web_md', 'core_web_md+c', 
                   'crunchbase_cased', 'crunchbase_uncased', 'date_detector', 'doc_history', 'full_name_detector', 'geo_cased', 
                   'geo_uncased', 'legal_detector', 'misc_detector', 'money_detector', 'number_detector', 'product_cased', 
                   'product_uncased', 'snips', 'time_detector', 'wiki_cased', 'wiki_small_cased']
            
        for source in sources:
            
            if source not in doc.user_data["annotations"]:
                continue
                
            for (start,end), vals in doc.user_data["annotations"][source].items():

                if (start,end) not in span_labels:
                    span_labels[(start,end)] = {}
                    
                for label, conf in vals:
                    span_labels[(start,end)][label] = span_labels[(start,end)].get(label, 0) + conf

                # We also look at overlapping spans (weighted by their overlap ratio)
                for start2, end2, vals2 in get_overlaps(start, end, doc.user_data["annotations"], sources):
                    if (start,end)!=(start2,end2):
                        overlap = (min(end,end2) - max(start,start2)) / (end-start)
                        for label2, conf2 in vals2:
                            span_labels[(start,end)][label2] = span_labels[(start,end)].get(label2, 0) + conf2*overlap  
        
        # We normalise
        for (start, end), label_counts in list(span_labels.items()):
            span_labels[(start,end)] = {label:count/sum(label_counts.values()) 
                                        for label, count in label_counts.items()}
 
        # We then count the label occurrences per entity string
        tokens = tuple(tok.text for tok in doc)
        for (start,end), weighted_labels in span_labels.items():
            span_string = tokens[start:end]
            if span_string in entity_counts:
                for label, label_weight in weighted_labels.items():
                    entity_counts[span_string][label] = entity_counts[span_string].get(label, 0) + label_weight
            else:
                entity_counts[span_string] = weighted_labels

        return entity_counts
    
    
    def add_annotations(self, doc, trie, case_sensitive=True):
        
        source = "doc_majority_%s"%("cased" if case_sensitive else "uncased")

        tokens = tuple(tok.text for tok in doc)
        for i in range(len(tokens)):
            span = tokens[i:i+8]
                
            prefix_length, label = trie.longest_prefix(span, case_sensitive)
            
            # We need to check whether the annotation does not overlap with itself
            if label:
                is_compatible = True
                for (start2,end2,label2) in get_overlaps(i, i+prefix_length, doc.user_data["annotations"], [source]):
                    
                    # If an overlap is detected, we select the longest span
                    if  end2-start2 < prefix_length:
                        del doc.user_data["annotations"][source][(start2,end2)]
                    else:
                        is_compatible=False
                        break
                if is_compatible:
                    self.add(doc, i, i+prefix_length, label, source)

 

    
    
############################################
# HELPER METHODS
############################################
  
    
def is_likely_proper(tok):
    """Returns true if the spacy token is a likely proper name, based on its form."""
    if len(tok)< 2:
        return False
    
    # If the lemma is titled, just return True
    elif tok.lemma_.istitle():
        return True
       
    # Handling cases such as iPad
    elif len(tok)>2 and tok.text[0].islower() and tok.text[1].isupper() and tok.text[2:].islower():
        return True
    
    elif (tok.is_upper and tok.text not in spacy_wrapper.CURRENCY_CODES 
          and tok.text not in spacy_wrapper.NOT_NAMED_ENTITIES):
        return True
    
    # Else, check whether the surface token is titled and is not sentence-initial
    elif (tok.i > 0 and tok.is_title and not tok.is_sent_start and tok.nbor(-1).text not in {'\'', '"', '‘', '“', '”', '’'} 
          and not tok.nbor(-1).text.endswith(".")):
        return True
    return False  
                           

def is_infrequent(span):  
    """Returns true if there is at least one token with a rank > 15000"""
    max_rank = max([tok2.rank if tok2.rank > 0 else 20001 for tok2 in span])
    return max_rank > 15000

def in_compound(tok):
    """Returns true if the spacy token is part of a compound phrase"""
    if tok.dep_=="compound":
        return True
    elif tok.i > 0 and tok.nbor(-1).dep_=="compound":
        return True
    return False


def get_spans(doc, sources, skip_overlaps=True):
    spans = set()
    for source in sources:
        if source not in doc.user_data["annotations"]:
            raise RuntimeError("Must run " + source + " first")
        for (start, end) in doc.user_data["annotations"][source]:
            spans.add((start,end))
    
    # If two spans are overlapping, return the longest spans
    finished = False   
    while skip_overlaps and not finished:
        finished = True
        sorted_spans = sorted(spans, key=lambda x:x[0])
        for (start1,end1),(start2,end2) in zip(sorted_spans[:-1], sorted_spans[1:]):
            if start2<end1:
                if (end1-start1) > (end2-start2):
                    spans.remove((start2,end2))
                else:
                    spans.remove((start1,end1))
                finished = False
                break
    return spans
    
 
def merge_contiguous_spans(spans, spacy_doc):
    """Merge spans that are contiguous (and with same label), or only separated with a comma"""
    
    
    finished = False   
    while not finished:
        finished = True
        sorted_spans = sorted(spans, key=lambda x:x[0])
        for (start1,end1),(start2,end2) in zip(sorted_spans[:-1], sorted_spans[1:]):
            if end1==start2 or (end1==start2-1 and spacy_doc[end1].text==","):
                val1 = spans[(start1, end1)]
                val2 = spans[start2, end2]
                if val1==val2:
                    del spans[(start1,end1)]
                    del spans[(start2,end2)]
                    spans[(start1,end2)] = val1
                    finished = False
                    break
    return spans


def get_overlaps(start, end, annotations, sources=None):
    """Returns a list of overlaps (as (start, end, value) between the provided span 
    and the existing annotations for the sources"""
    
    overlaps = []
    for source in (sources if sources is not None else annotations.keys()):
        intervals = list(annotations[source].keys())
        
        start_search, end_search = _binary_search(start, end, intervals)
        
        for interval_start, interval_end in intervals[start_search:end_search]:
            if start < interval_end and end > interval_start:
                interval_value = annotations[source][(interval_start, interval_end)]
                overlaps.append((interval_start, interval_end, interval_value))
            
    return overlaps
   

def _binary_search(start, end, intervals):
    """Performs a binary search"""
    start_search = 0
    end_search = len(intervals)
    while start_search < (end_search-1):        
        mid = start_search + (end_search-start_search)//2
        (interval_start, interval_end) = intervals[mid]
        
        if interval_end <= start:
            start_search = mid
        elif interval_start >= end:
            end_search = mid
        else:
            break
    return start_search, end_search



############################################
# VI
############################################
  
    
def display_entities(spacy_doc, source=None):
    """Display the entities in a spacy document (with some preprocessing to handle special characters)"""
    
    if source is None:
        spans = [(ent.start,ent.end, ent.label_) for ent in spacy_doc.ents]
    else:
        spans = [(start,end, label) for start,end in spacy_doc.user_data["annotations"][source] 
                 for label, conf in spacy_doc.user_data["annotations"][source][(start,end)] if conf > 0.2]
        
    text = spacy_doc.text
    # Due to a rendering bug, we need to escape dollar signs, and keep the offsets to get the 
    # entity boundaries right in respect to the text
    dollar_sign_offsets = [i for i in range(len(text)) if text[i]=="$"]
    text = text.replace("$", "\$")
    
    entities = {}
    for start,end, label in spans:
        
        start_char = spacy_doc[start].idx 
        end_char = spacy_doc[end-1].idx + len(spacy_doc[end-1])
        
        # We need to pad the character offsets for escaped dollar signs
        if dollar_sign_offsets:
            start_char += np.searchsorted(dollar_sign_offsets, start_char)
            end_char += np.searchsorted(dollar_sign_offsets, end_char)
            
        if (start_char,end_char) not in entities:
            entities[(start_char, end_char)] = label
            
        # If we have several alternative labels for a span, join them with +
        elif label not in entities[(start_char,end_char)]:
            entities[(start_char,end_char)] = entities[(start_char,end_char)]+ "+" + label
            
    entities = [{"start":start, "end":end, "label":label} for (start,end), label in entities.items()]
    doc2 = {"text":text, "title":None, "ents":entities}
    spacy.displacy.render(doc2, jupyter=True, style="ent", manual=True)

 

############################################
# DATA CRUNCHING
############################################

    
def compile_wikidata(wikidata="../data/WikidataNE_20170320_NECKAR_1_0.json_.gz", only_with_descriptions=False):
    """Compiles a JSON file with the wiki data"""
     
    
    import gzip, json
    fd = gzip.open(wikidata)
    wikidata = {"PERSON":{}, "LOC":{}, "GPE":{}, "ORG":{}}
    location_qs = set()
    for l in fd:
        d = json.loads(l)
        neClass = str(d["neClass"])
        name = d["norm_name"]
        if ("en_sitelink" not in d and neClass !="PER"):
            continue
        if "en_sitelink" in d:
            if "," in d["en_sitelink"] or "(" in d["en_sitelink"]:
                continue
        if name[0].isdigit() or name[-1].isdigit() or len(name) < 2:
            continue
        if neClass=="PER":
            neClass = "PERSON"
        elif neClass=="LOC":
            if {'Mountain Range', 'River', 'Sea', 'Continent', 'Mountain'}.intersection(d.get("location_type",set())):
                neClass = "LOC"
            else:
                neClass ="GPE"
            location_qs.add(d["id"])
        elif neClass=="ORG" and d["id"] in location_qs:
            continue
        if "alias" in d:
            d["nb_aliases"] = len(d["alias"])
            del d["alias"]
        for key_to_remove in ["de_sitelink", '$oid', "id", "coordinate", "official_website", "_id"]:
            if key_to_remove in d:
                del d[key_to_remove]
        if name in wikidata[neClass]:
            merge = wikidata[neClass][name] if len(str(wikidata[neClass][name])) > len(str(d)) else d
            merge["nb_entities"] = wikidata[neClass][name].get("nb_entities", 1) + 1
            wikidata[neClass][name] = merge
        else:
            wikidata[neClass][name] = d
                  
    fd = open("data/frequencies.pkl", "rb")
    frequencies = pickle.load(fd)
    fd.close() 
    
    # We only keep entities with a certain frequency
    for neClass in ["PERSON", "LOC", "ORG", "GPE"]:
        for entity in list(wikidata[neClass].keys()): 
            if entity.lower() in frequencies and frequencies[entity.lower()]>10000: 
                del wikidata[neClass][entity]
    
    # And prune those that cannot be encoded using latin characters
    for neClass in ["PERSON", "LOC", "ORG", "GPE"]:
        for entity in list(wikidata[neClass].keys()): 
            try:
                entity.encode('iso-8859-15') 
            except UnicodeEncodeError: 
                del wikidata[neClass][entity]

        
    wikidata2 = {neClass:{} for neClass in wikidata}
    for neClass in wikidata:
        entities_for_class = set()
        for entity in wikidata[neClass]:
            nb_tokens = len(entity.split())
            if nb_tokens > 10:
                continue
            if only_with_descriptions and "description" not in wikidata[neClass][entity]:
                continue
            entities_for_class.add(entity) 
            if "en_sitelink" in wikidata[neClass][entity]:
                entities_for_class.add(wikidata[neClass][entity]["en_sitelink"])
        wikidata2[neClass] = entities_for_class
                            
    fd = open(WIKIDATA_SMALL if only_with_descriptions else WIKIDATA, "w")
    json.dump({key:sorted(names) for key,names in wikidata2.items()}, fd)
    fd.close()

    
def get_alternative_company_names(name, vocab=None):
    """Extract a list of alternative company names (with or without legal suffix etc.)"""
    
    alternatives = {name}        
    while True:
        current_nb_alternatives = len(alternatives)
            
        for alternative in list(alternatives):
            tokens = alternative.split()
            if len(tokens)==1:
                continue
                
            # We add an alternative name without the legal suffix
            if tokens[-1].lower().rstrip(".") in spacy_wrapper.LEGAL_SUFFIXES: 
                alternatives.add(" ".join(tokens[:-1]))
            
            if tokens[-1].lower() in {"limited", "corporation"}:
                alternatives.add(" ".join(tokens[:-1]))                
                
            if tokens[-1].lower().rstrip(".") in {"corp", "inc", "co"}:
                if alternative.endswith("."):
                    alternatives.add(alternative.rstrip("."))
                else:
                    alternatives.add(alternative+".")
                
            # If the last token is a country name (like The SAS Group Norway), add an alternative without
            if tokens[-1] in COUNTRIES:
                alternatives.add(" ".join(tokens[:-1])) 
                
            # If the name starts with a the, add an alternative without it
            if tokens[0].lower()=="the":   
                alternatives.add(" ".join(tokens[1:]))
                
            # If the name ends with a generic token such as "Telenor International", add an alternative without
            if vocab is not None and tokens[-1] in GENERIC_TOKENS and any([tok for tok in tokens if vocab[tok].rank==0]):
                alternatives.add(" ".join(tokens[:-1])) 
                    
        if len(alternatives)==current_nb_alternatives:
            break
    
    # We require the alternatives to have at least 2 characters (4 characters if the name does not look like an acronym)
    alternatives = {alt for alt in alternatives if len(alt) > 1 and alt.lower().rstrip(".") not in spacy_wrapper.LEGAL_SUFFIXES} 
    alternatives = {alt for alt in alternatives if len(alt) > 3 or alt.isupper()}
    
    return alternatives


    
def compile_geographical_data(geo_source="../data/allCountries.txt", population_threshold=100000):
    """Compiles a JSON file with geographical locations"""
    
    names = set()
    fd = open(geo_source)
    for i, line in enumerate(fd):
        line_feats = line.split("\t")
        if len(line_feats) < 15:
            continue
        population = int(line_feats[14])
        if population < population_threshold:
            continue
        name = line_feats[1].strip()
        names.add(name)
        name = re.sub(".*(?:Kingdom|Republic|Province|State|Commonwealth|Region|City|Federation) of ", "", name).strip()
        names.add(name)
        name = name.replace(" City", "").replace(" Region", "").replace(" District", "").replace(" County", "").replace(" Zone", "").strip()
        names.add(name)
        name = (name.replace("Arrondissement de ", "").replace("Stadtkreis ", "").replace("Landkreis ", "").strip()
                .replace("Departamento de ", "").replace("Département de ", "").replace("Provincia di ", "")).strip()
        names.add(name)
        name = re.sub("^the ", "", name).strip()
        names.add(name)
        if i%10000==0:
            print("Number of processed lines:", i, "and number of extracted locations:", len(names))
    fd.close()
    names = {alt for alt in names if len(alt) > 2 and alt.lower().rstrip(".") not in spacy_wrapper.LEGAL_SUFFIXES}
    fd = open(GEONAMES, "w")
    json.dump({"GPE":sorted(names)}, fd)
    fd.close()
        
        
def compile_crunchbase_data(org_data="../data/organizations.csv", people_data="../data/people.csv"):
    """Compiles a JSON file with company and person names from Crunchbase Open Data"""

    company_entities = set()
    other_org_entities = set()
    
    vocab = spacy.load("en_core_web_md").vocab
    
    fd = open(org_data)
    for line in fd:
        split = [s.strip() for s in line.rstrip().strip("\"").split("\",\"")]
        if len(split) < 5:
            continue
        name = split[1]
        alternatives = get_alternative_company_names(name, vocab)        
        if split[3] in {"company", "investor"}:
            company_entities.update(alternatives)
        else:
            other_org_entities.update(alternatives)
    fd.close()
    print("Number of extracted entities: %i companies and %i other organisations"%(len(company_entities), len(other_org_entities)))

    persons = set()
    fd = open(people_data)
    for line in fd:
        split = [s.strip() for s in line.rstrip().strip("\"").split("\",\"")]
        if len(split) < 5:
            continue
        first_name = split[2]
        last_name = split[3]
        alternatives = {"%s %s"%(first_name, last_name)}
    #    alternatives.add(last_name)
        alternatives.add("%s. %s"%(first_name[0], last_name))
        if " " in first_name:
            first_split = first_name.split(" ", 1)
            alternatives.add("%s %s"%(first_split[0], last_name))
            alternatives.add("%s %s. %s"%(first_split[0], first_split[1][0], last_name))
            alternatives.add("%s. %s. %s"%(first_split[0][0], first_split[1][0], last_name))
        persons.update(alternatives)
        
    # We require person names to have at least 3 characters (and not be a suffix)
    persons = {alt for alt in persons if len(alt) > 2 and alt.lower().rstrip(".") not in spacy_wrapper.LEGAL_SUFFIXES}
    fd.close()
    print("Number of extracted entities: %i person names"%(len(persons)))
   
    fd = open(CRUNCHBASE, "w")
    json.dump({"COMPANY":sorted(company_entities), "ORG":sorted(other_org_entities), "PERSON":sorted(persons)}, fd)
    fd.close()
    
def compile_product_data(data="../data/dbpedia.json"):
    fd = open(data)
    all_product_names = set()
    for line in fd:
        line = line.strip().strip(",")
        value = json.loads(line)["label2"]["value"]
        if "(" in value:
            continue
            
        product_names = {value}
        
        # The DBpedia entries are all titled, which cause problems for products such as iPad
        if len(value)>2 and value[0] in {"I", "E"} and value[1].isupper() and value[2].islower():
            product_names.add(value[0].lower()+value[1:])
        
        # We also add plural entries
        for product_name in list(product_names):
            if len(product_name.split()) <= 2:
                plural = product_name + ("es" if value.endswith("s") else "s")
                product_names.add(plural)
                
        all_product_names.update(product_names)
        
    fd = open(PRODUCTS, "w")
    json.dump({"PRODUCT":sorted(all_product_names)}, fd)
    fd.close()
        
        
def compile_wiki_product_data(data="../data/wiki_products.json"):
    fd = open(data)
    dict_list = json.load(fd)
    fd.close()
    products = set()
    for product_dict in dict_list:
        product_name = product_dict["itemLabel"]
        if "("  in product_name or len(product_name) <= 2:
            continue
        products.add(product_name)
        if len(product_name.split()) <= 2:
            plural = product_name + ("es" if product_name.endswith("s") else "s")
            products.add(plural)

    fd = open(WIKIDATA, "r")
    current_dict = json.load(fd)
    fd.close()
    current_dict["PRODUCT"] = sorted(products)
    fd = open(WIKIDATA, "w")
    json.dump(current_dict, fd)
    fd.close()
    
    fd = open(WIKIDATA_SMALL, "r")
    current_dict = json.load(fd)
    fd.close()
    current_dict["PRODUCT"] = sorted(products)
    fd = open(WIKIDATA_SMALL, "w")
    json.dump(current_dict, fd)
    fd.close()
