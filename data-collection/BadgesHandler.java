import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

import javax.xml.XMLConstants;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;
import java.io.IOException;

class BadgeRow {
    private String Id;
    private String UserId;
    private String Name;
    private String Date;
    private String BadgeClass;
    private String TagBased;

    public BadgeRow(String id, String userId, String name, String date, String badgeClass, String tagBased) {
        Id = id;
        UserId = userId;
        Name = name;
        Date = date;
        BadgeClass = badgeClass;
        TagBased = tagBased;
    }

    public String getId() {
        return Id;
    }

    public String getUserId() {
        return UserId;
    }

    public String getName() {
        return Name;
    }

    public String getDate() {
        return Date;
    }

    public String getBadgeClass() {
        return BadgeClass;
    }

    public String getTagBased() {
        return TagBased;
    }

    @Override
    public String toString() {
        return "BadgeRow{" +
                "Id='" + Id + '\'' +
                ", UserId='" + UserId + '\'' +
                ", Name='" + Name + '\'' +
                ", Date='" + Date + '\'' +
                ", BadgeClass='" + BadgeClass + '\'' +
                ", TagBased='" + TagBased + '\'' +
                '}';
    }
}

public class BadgesHandler extends DefaultHandler {
    private static final String ROW = "row";
    private static final String BADGES = "badges";

    @Override
    public void startElement(String uri, String lName, String qName, Attributes attr) throws SAXException {
        if (qName.equals(ROW)) {
            BadgeRow badgeRow = new BadgeRow(
                    attr.getValue("Id"),
                    attr.getValue("UserId"),
                    attr.getValue("Name"),
                    attr.getValue("Date"),
                    attr.getValue("Class"),
                    attr.getValue("TagBased")
            );
            System.out.println(badgeRow);

        } else if (!qName.equals(BADGES)) {
            throw new SAXException(String.format("Unknown tag %s", qName));
        }

    }

    public static void main(String[] args) throws ParserConfigurationException, SAXException, IOException {
        final long start = System.currentTimeMillis();
        System.out.println(start);
        SAXParserFactory factory = SAXParserFactory.newInstance();
        factory.setFeature(XMLConstants.FEATURE_SECURE_PROCESSING,false);
        SAXParser saxParser = factory.newSAXParser();
        BadgesHandler handler = new BadgesHandler();
        saxParser.parse(
                "..\\data\\raw\\stackoverflow.com-Badges\\badges.xml", handler);


        // time end
        final long durationInMilliseconds = System.currentTimeMillis()-start;
        System.out.println("SAX parse took " + durationInMilliseconds + "ms.");
    }
}
