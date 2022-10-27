import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

import javax.xml.XMLConstants;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;
import java.io.IOException;

class TagRow {
    private String Id;
    private String TagName;
    private String Count;

    public TagRow(String Id, String TagName, String Count) {
        this.Id = Id;
        this.TagName = TagName;
        this.Count = Count;
    }

    public String getId() {
        return Id;
    }

    public String getTagName() {
        return TagName;
    }

    public String getCount() {
        return Count;
    }

    @Override
    public String toString() {
        return "TagRow{" +
                "Id='" + Id + '\'' +
                ", TagName='" + TagName + '\'' +
                ", Count='" + Count + '\'' +
                '}';
    }
}

    public class TagsHandler extends DefaultHandler {
    private static final String ROW = "row";
    private static final String TAGS = "tags";

    @Override
    public void startElement(String uri, String lName, String qName, Attributes attr) throws SAXException {
        if (qName.equals(ROW)) {
            TagRow badgeRow = new TagRow(
                    attr.getValue("Id"),
                    attr.getValue("TagName"),
                    attr.getValue("Count")
            );
            System.out.println(badgeRow);

        } else if (!qName.equals(TAGS)) {
            throw new SAXException(String.format("Unknown tag %s", qName));
        }

    }

    public static void main(String[] args) throws ParserConfigurationException, SAXException, IOException {
        final long start = System.currentTimeMillis();
        System.out.println(start);
        SAXParserFactory factory = SAXParserFactory.newInstance();
        factory.setFeature(XMLConstants.FEATURE_SECURE_PROCESSING,false);
        SAXParser saxParser = factory.newSAXParser();
        TagsHandler handler = new TagsHandler();
        saxParser.parse(
                "..\\data\\raw\\stackoverflow.com-Tags\\tags.xml", handler);


        // time end
        final long durationInMilliseconds = System.currentTimeMillis()-start;
        System.out.println("SAX parse took " + durationInMilliseconds + "ms.");
    }
}
